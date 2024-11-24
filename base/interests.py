ANNUAL_INTEREST_RATES = {3: 0.0215, 5: 0.02}


def calculate_interest(
    deposit: float = 170000,
    term_years: int = 3,
    has_extra_sufficient_funds: bool = False,
) -> float:
    """
    Calculate the total interest for a given deposit and term.

    Args:
        deposit (float): The initial deposit amount.
        term_years (int): The term of the deposit in years.
        has_extra_sufficient_funds (bool): Whether there are extra sufficient funds.

    Returns:
        float: The total interest earned.
    """
    if term_years not in ANNUAL_INTEREST_RATES:
        raise ValueError(f"Term of {term_years} years is not supported.")

    annual_rate = ANNUAL_INTEREST_RATES[term_years]

    # If there are extra sufficient funds, simply calculate the interest based on the annual rate
    if has_extra_sufficient_funds:
        total_interest = deposit * annual_rate * term_years
        print(
            f"Deposit {deposit:.0f}, {term_years}-year interest with extra sufficient funds: {total_interest:.2f}"
        )
    else:  # Otherwise, calculate the interest by breaking it down into monthly payments
        monthly_rate = annual_rate / 12
        monthly_payment = deposit / (term_years * 12)
        total_interest = 0

        for profitable_months in range(1, term_years * 12 + 1):
            total_interest += monthly_payment * monthly_rate * profitable_months

        print(
            f"Deposit {deposit:.0f}, {term_years}-year interest: {total_interest:.2f}"
        )

    return total_interest


if __name__ == "__main__":
    calculate_interest()
    calculate_interest(has_extra_sufficient_funds=True)

    calculate_interest(term_years=5)
    calculate_interest(term_years=5, has_extra_sufficient_funds=True)
