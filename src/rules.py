def generate_rule(income, credit):

    if income < 100000 and credit > 500000:
        return "HIGH RISK"

    elif income < 200000:
        return "MEDIUM RISK"

    else:
        return "LOW RISK"


print(generate_rule(80000,600000))