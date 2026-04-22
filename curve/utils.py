from sklearn.isotonic import IsotonicRegression


def check_call_spread_arbitrage(strikes, prices):
    """
    Call spread arbitrage 체크
    포지션 비용: C(K1) - C(K2)
    max payoff: (S-k1 - C(K1)) - (S-K2 - C(K2)) >= 0
    - C(K1) - C(K2) <= K2 - K1
    """
    violations = []
    for i in range(len(strikes) - 1):
        cost = prices[i] - prices[i + 1]
        max_payoff = strikes[i + 1] - strikes[i]
        if cost > max_payoff:
            violations.append({
                'K1': strikes[i],
                'K2': strikes[i + 1],
                'cost': cost,
                'max_payoff': max_payoff,
                'violation': cost - max_payoff
            })
    return violations


def check_butterfly_arbitrage(strikes, prices):
    """
    Butterfly arbitrage 체크
    - C(K1) - 2*C(K2) + C(K3) >= 0
    - PDF > 0 조건과 동일 (이산화버전)
    """
    violations = []
    for i in range(len(strikes) - 2):
        if strikes[i + 1] - strikes[i] == strikes[i + 2] - strikes[i + 1]:
            cost = prices[i] - 2 * prices[i + 1] + prices[i + 2]
            if cost < -1e-6:
                violations.append({
                    'K': [strikes[i], strikes[i + 1], strikes[i + 2]],
                    'cost': cost
                })
    return violations


def clean_arbitrage_prices(strikes, prices, max_iter=10):
    # Call price는 K에 대해 감소해야 함
    iso_reg = IsotonicRegression(increasing=False)
    prices_clean = iso_reg.fit_transform(strikes, prices)

    # 2차 차분이 양수여야 함 (convexity 조건)
    for _ in range(max_iter):
        violations = 0
        for i in range(1, len(strikes) - 1):
            d2C = prices_clean[i - 1] - 2 * prices_clean[i] + prices_clean[i + 1]
            if d2C < 0:
                alpha = (strikes[i] - strikes[i - 1]) / (strikes[i + 1] - strikes[i - 1])
                prices_clean[i] = (1 - alpha) * prices_clean[i - 1] + alpha * prices_clean[i + 1]
                violations += 1
        if violations == 0:
            break

    return prices_clean
