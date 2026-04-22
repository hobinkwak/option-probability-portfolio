import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gbs import GBSEquation

warnings.filterwarnings('ignore')


class OptionChainMonitor:
    """
    옵션 체인 Greek Exposure 모니터링
    """

    def __init__(self, data, rf, dividend, contract_size=100):
        self.oc_raw = data.copy()
        self.oc_raw['T'] = self.oc_raw['T'].clip(lower=6.5 / 24 / 365)
        self.S = data['UNDERLYING_LAST'].values[0]
        self.r = rf
        self.d = dividend
        self.contract_size = contract_size

        self.gex = None
        self.vex = None
        self.gamma_profile = None
        self.vanna_profile = None
        self.zero_gamma = None
        self.zero_vanna = None

        self._preprocess()
        self._ensure_greeks()

    def _preprocess(self):
        call_cols = ['STRIKE', 'C_IV', 'C_VOLUME', 'C_LAST', 'C_BID', 'C_ASK', 'C_MID', 'C_OI', 'T']
        call_greeks = [c for c in ['C_DELTA', 'C_GAMMA', 'C_VEGA', 'C_THETA'] if c in self.oc_raw.columns]
        self.call = self.oc_raw[call_cols + call_greeks].dropna(subset=['C_IV', 'C_MID'])

        put_cols = ['STRIKE', 'P_IV', 'P_VOLUME', 'P_LAST', 'P_BID', 'P_ASK', 'P_MID', 'P_OI', 'T']
        put_greeks = [c for c in ['P_DELTA', 'P_GAMMA', 'P_VEGA', 'P_THETA'] if c in self.oc_raw.columns]
        self.put = self.oc_raw[put_cols + put_greeks].dropna(subset=['P_IV', 'P_MID'])

    def _ensure_greeks(self):
        # IV가 있으면 gamma, vega, vanna를 한번에 계산
        for prefix, df, iv_col in [('C', self.call, 'C_IV'), ('P', self.put, 'P_IV')]:
            K, T, iv = df['STRIKE'], df['T'], df[iv_col]
            gbs = GBSEquation(self.S, K, T, self.r, iv, self.d)

            if f'{prefix}_GAMMA' not in df.columns:
                self.call[f'{prefix}_GAMMA'] = gbs.gamma() if prefix == 'C' else None
                self.put[f'{prefix}_GAMMA'] = gbs.gamma() if prefix == 'P' else None

            if f'{prefix}_VEGA' not in df.columns:
                self.call[f'{prefix}_VEGA'] = gbs.vega() if prefix == 'C' else None
                self.put[f'{prefix}_VEGA'] = gbs.vega() if prefix == 'P' else None

            if f'{prefix}_VANNA' not in df.columns:
                vanna = gbs.vanna()
                vanna = np.where(np.isinf(vanna), np.nan, vanna)
                if prefix == 'C':
                    self.call[f'{prefix}_VANNA'] = vanna
                else:
                    self.put[f'{prefix}_VANNA'] = vanna

    def _get_boundary(self, boundary=None):
        if boundary is not None:
            return boundary
        ks_ratio = self.call['STRIKE'] / self.S - 1
        return min(abs(ks_ratio.min()), abs(ks_ratio.max()))

    def _clip_by_boundary(self, df, boundary):
        lo = self.S * (1 - boundary)
        hi = self.S * (1 + boundary)
        return df.loc[(df.index >= lo) & (df.index <= hi)]

    @staticmethod
    def _find_zero_crossings(values, levels):
        sign_changes = np.where(np.diff(np.sign(values)))[0]
        zeros = []
        for idx in sign_changes:
            v0, v1 = values[idx], values[idx + 1]
            k0, k1 = levels[idx], levels[idx + 1]
            if v1 != v0:
                zeros.append(k0 - v0 * (k1 - k0) / (v1 - v0))
        return zeros

    def compute_gex(self, boundary=None):
        # Gamma Exposure (dealer short put, long call convention)
        cs = self.contract_size
        S2 = (self.S ** 2) / 100

        self.call = self.call.copy()
        self.put = self.put.copy()
        self.call['C_GEX'] = self.call['C_GAMMA'] * S2 * cs * self.call['C_OI']
        self.put['P_GEX'] = self.put['P_GAMMA'] * S2 * cs * self.put['P_OI'] * (-1)

        gex = pd.concat([self.call['C_GEX'], self.put['P_GEX']], axis=1).dropna(how='all')
        gex.index = self.oc_raw.loc[gex.index, 'STRIKE']
        gex = gex.groupby(gex.index).sum()
        gex['NET'] = gex.sum(axis=1)

        boundary = self._get_boundary(boundary)
        self.gex = self._clip_by_boundary(gex, boundary)
        return self.gex

    def compute_vex(self, boundary=None):
        # Vanna Exposure
        self.call = self.call.copy()
        self.put = self.put.copy()
        self.call['C_VEX'] = self.call['C_VANNA'] * self.S * self.call['C_IV'] * self.call['C_OI'] * (-1)
        self.put['P_VEX'] = self.put['P_VANNA'] * self.S * self.put['P_IV'] * self.put['P_OI']

        vex = pd.concat([self.call['C_VEX'], self.put['P_VEX']], axis=1).dropna(how='all')
        vex.index = self.oc_raw.loc[vex.index, 'STRIKE']
        vex = vex.groupby(vex.index).sum()
        vex['NET'] = vex.sum(axis=1)

        boundary = self._get_boundary(boundary)
        self.vex = self._clip_by_boundary(vex, boundary)
        return self.vex

    def compute_gamma_profile(self, boundary=None, n_levels=1000):
        boundary = self._get_boundary(boundary)
        cs = self.contract_size
        levels = np.linspace(self.S * (1 - boundary), self.S * (1 + boundary), n_levels)

        # 벡터화: levels (n_levels,) x strikes (n_strikes,) → (n_levels, n_strikes)
        c_K = self.call['STRIKE'].values
        c_T = self.call['T'].values
        c_IV = self.call['C_IV'].values
        c_OI = self.call['C_OI'].values

        p_K = self.put['STRIKE'].values
        p_T = self.put['T'].values
        p_IV = self.put['P_IV'].values
        p_OI = self.put['P_OI'].values

        gammas = np.empty(n_levels)
        for i, level in enumerate(levels):
            c_g = GBSEquation(level, c_K, c_T, self.r, c_IV, self.d).gamma()
            p_g = GBSEquation(level, p_K, p_T, self.r, p_IV, self.d).gamma()
            c_gex = np.nansum(c_g * ((level ** 2) / 100) * cs * c_OI)
            p_gex = np.nansum(p_g * ((level ** 2) / 100) * cs * p_OI * (-1))
            gammas[i] = c_gex + p_gex

        self.gamma_profile = pd.Series(gammas, index=levels)

        zeros = self._find_zero_crossings(gammas, levels)
        self.zero_gamma = zeros[0] if zeros else None

        return self.gamma_profile, self.zero_gamma

    def compute_vanna_profile(self, boundary=None, n_levels=1000):
        boundary = self._get_boundary(boundary)
        levels = np.linspace(self.S * (1 - boundary), self.S * (1 + boundary), n_levels)

        c_K = self.call['STRIKE'].values
        c_T = self.call['T'].values
        c_IV = self.call['C_IV'].values
        c_OI = self.call['C_OI'].values

        p_K = self.put['STRIKE'].values
        p_T = self.put['T'].values
        p_IV = self.put['P_IV'].values
        p_OI = self.put['P_OI'].values

        vannas = np.empty(n_levels)
        for i, level in enumerate(levels):
            c_v = GBSEquation(level, c_K, c_T, self.r, c_IV, self.d).vanna()
            c_v = np.where(np.isinf(c_v), 0.0, np.nan_to_num(c_v))
            p_v = GBSEquation(level, p_K, p_T, self.r, p_IV, self.d).vanna()
            p_v = np.where(np.isinf(p_v), 0.0, np.nan_to_num(p_v))

            c_vex = np.nansum(c_v * level * c_IV * c_OI * (-1))
            p_vex = np.nansum(p_v * level * p_IV * p_OI)
            vannas[i] = c_vex + p_vex

        self.vanna_profile = pd.Series(vannas, index=levels)

        zeros = self._find_zero_crossings(vannas, levels)
        self.zero_vanna = zeros[0] if zeros else None

        return self.vanna_profile, self.zero_vanna

    def visualize_gex(self, boundary=None, ax=None, bar_width=None, save_path='plot/'):
        if self.gex is None:
            self.compute_gex(boundary)

        boundary = self._get_boundary(boundary)
        gex = self._clip_by_boundary(self.gex, boundary)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        if bar_width is None:
            strikes = gex.index.values
            diffs = np.diff(strikes)
            bar_width = np.median(diffs[diffs > 0]) * 2.5 if len(strikes) > 1 else 50

        colors = np.where(gex['NET'] >= 0, '#2166ac', '#d6604d')
        ax.bar(gex.index, gex['NET'], color=colors, width=bar_width, alpha=1.0, edgecolor='none')

        ax.axvline(self.S, color='k', linestyle='--', linewidth=1, label=f'Spot ({self.S:,.0f})')
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Strike')
        ax.set_ylabel('Gamma Exposure (GEX)')
        ax.set_title('Net Gamma Exposure by Strike')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}GEX.png')
        plt.show()

    def visualize_vex(self, boundary=None, ax=None, bar_width=None, save_path='plot/'):
        if self.vex is None:
            self.compute_vex(boundary)

        boundary = self._get_boundary(boundary)
        vex = self._clip_by_boundary(self.vex, boundary)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        if bar_width is None:
            strikes = vex.index.values
            bar_width = np.median(np.diff(strikes)) * 0.8 if len(strikes) > 1 else 50

        colors = np.where(vex['NET'] >= 0, 'steelblue', 'salmon')
        ax.bar(vex.index, vex['NET'], color=colors, width=bar_width, alpha=0.8)
        ax.axvline(self.S, color='k', linestyle='--', linewidth=1, label=f'Spot ({self.S:,.0f})')
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Strike')
        ax.set_ylabel('Vanna Exposure (VEX)')
        ax.set_title('Net Vanna Exposure by Strike')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}VEX.png')
        plt.show()

    def visualize_gamma_profile(self, boundary=None, ax=None, save_path='plot/'):
        if self.gamma_profile is None:
            self.compute_gamma_profile(boundary)

        boundary = self._get_boundary(boundary)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        profile = self.gamma_profile.loc[
            (self.gamma_profile.index >= self.S * (1 - boundary)) &
            (self.gamma_profile.index <= self.S * (1 + boundary))
            ]
        ax.plot(profile.index, profile.values, color='steelblue', linewidth=1.5)
        ax.fill_between(profile.index, profile.values, 0,
                        where=profile.values >= 0, color='steelblue', alpha=0.15)
        ax.fill_between(profile.index, profile.values, 0,
                        where=profile.values < 0, color='salmon', alpha=0.15)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(self.S, color='red', linestyle='--', linewidth=1, label=f'Spot ({self.S:,.0f})')
        if self.zero_gamma is not None:
            ax.axvline(self.zero_gamma, color='green', linestyle='--', linewidth=1,
                       label=f'Gamma Flip ({self.zero_gamma:,.0f})')
        ax.set_xlabel('Price Level')
        ax.set_ylabel('Net Gamma Exposure')
        ax.set_title('Gamma Profile')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}Gamma Profile.png')
        plt.show()

    def visualize_vanna_profile(self, boundary=None, ax=None, save_path='plot/'):
        if self.vanna_profile is None:
            self.compute_vanna_profile(boundary)

        boundary = self._get_boundary(boundary)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        profile = self.vanna_profile.loc[
            (self.vanna_profile.index >= self.S * (1 - boundary)) &
            (self.vanna_profile.index <= self.S * (1 + boundary))
            ]
        ax.plot(profile.index, profile.values, color='darkorange', linewidth=1.5)
        ax.fill_between(profile.index, profile.values, 0,
                        where=profile.values >= 0, color='steelblue', alpha=0.15)
        ax.fill_between(profile.index, profile.values, 0,
                        where=profile.values < 0, color='salmon', alpha=0.15)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(self.S, color='red', linestyle='--', linewidth=1, label=f'Spot ({self.S:,.0f})')
        if self.zero_vanna is not None:
            ax.axvline(self.zero_vanna, color='green', linestyle='--', linewidth=1,
                       label=f'Vanna Flip ({self.zero_vanna:,.0f})')
        ax.set_xlabel('Price Level')
        ax.set_ylabel('Net Vanna Exposure')
        ax.set_title('Vanna Profile')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}Vanna Profile.png')
        plt.show()


if __name__ == '__main__':
    pass
