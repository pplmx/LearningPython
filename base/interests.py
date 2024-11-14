ANNUAL_INTEREST_RATES = {3: 0.0215, 5: 0.02}


def calculate_interest(deposit=170000, term_years=3, has_extra_sufficient_funds=False):
    annual_rate = ANNUAL_INTEREST_RATES[term_years]

    if has_extra_sufficient_funds:
        # 如果有额外充足资金，简单按年利率计算
        return deposit * annual_rate * term_years

    # 否则，按月分期计算
    monthly_rate = annual_rate / 12
    monthly_payment = deposit / (term_years * 12)
    total_interest = 0

    for profitable_months in range(1, term_years * 12 + 1):
        total_interest += monthly_payment * monthly_rate * profitable_months

    return total_interest


if __name__ == '__main__':
    my_deposit = 170000

    print(f"3-year interest: {calculate_interest(my_deposit, 3):.2f}")
    print(f"3-year interest with extra sufficient funds: {calculate_interest(my_deposit, 3, True):.2f}")
    print(f"5-year interest: {calculate_interest(my_deposit, 5):.2f}")
    print(f"5-year interest with extra sufficient funds: {calculate_interest(my_deposit, 5, True):.2f}")
