"""
Home buying expenses and loan calculations.
"""

import logging
from collections.abc import Mapping, MutableMapping, Sequence
from copy import copy
from inspect import signature
from typing import Any, NewType, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize

__author__ = "Justin L. Lanfranchi"
__copyright__ = "Copyright (c) 2026, Justin L. Lanfranchi"
__license__ = "MIT"


__all__ = [
    "Params",
    "Info",
    "make_title",
    "make_headline",
    "compounding",
    "annual_to_monthly_rate_pct",
    "calc_payment_for_payoff",
    "details_and_pie_charts",
    "print_info",
    "generate_report",
    "merge_params",
    "extract_params_from_info",
    "make_dataframe",
    "evaluate",
    "quadratic_cost",
    "generic_optimizer",
    "brute_force_optimizer",
    "calc_down_payment_from_cash_at_closing",
    "calc_down_payment_for_monthly_cost",
]


logger = logging.getLogger(__name__)

# Number = NewType("Number", Union(float, int))
# NumericArray = NewType("NumericArray", npt.NDArray)
Params = NewType("Params", Mapping[str, float])
Info = NewType("Info", MutableMapping[str, float | Mapping[str, float] | Sequence[str]])


def make_title(title: str):
    """Print nice title

    Parameters
    ----------
    title

    """

    print("=" * 120)
    print(title.center(120))
    print("=" * 120)


def make_headline(headline: str):
    """Print nice headline (lesser than title).

    Parameters
    ----------
    headline

    """

    print()
    print(f" {headline} ".center(80, "-"))
    print()


def compounding(
    principal: float | npt.NDArray,
    interest_rate_pct: float | npt.NDArray,
    payment: float | npt.NDArray,
    num_periods: float | npt.NDArray,
) -> tuple[float | npt.NDArray, float | npt.NDArray, float | npt.NDArray]:
    """Compute values given discrete compounding.

    Parameters
    ----------
    principal
        starting balance due
    interest_rate_pct
        percent
    payment
        fixed payment per period
    num_periods
        number of periods to compute compounding & payments

    Returns
    -------
    balance
        total remaining owed balance after `num_periods`
    total_interest_payments
        total paid in interest
    total_payments
        total paid including both principal and interest

    """

    bcast = np.broadcast(principal, interest_rate_pct, payment, num_periods)

    interest_rate = interest_rate_pct / 100
    ihat = 1 + interest_rate
    if bcast.ndim == 0:  # all arguments are scalar
        # Special case if interest_rate is 0, as other equation has
        # interest_rate in the denominator (0/0). Otherwise, use the full
        # equation.
        if interest_rate == 0:
            balance = np.clip(
                principal - payment * num_periods,
                a_max=principal,
                a_min=0,
            )
        else:
            balance = np.clip(
                principal * ihat**num_periods
                - payment / interest_rate * (ihat**num_periods - 1),
                a_max=np.inf,
                a_min=0,
            )
    else:  # one or more arguments is an array (of any dimensionality)
        balance = np.empty(shape=bcast.shape)
        # 1. Compute those elements for which interest rate is 0 (divide by 0
        #    otherwise)
        where_zero = interest_rate == 0
        balance[where_zero] = np.clip(
            ((principal - payment) * num_periods)[where_zero],
            a_max=np.inf,
            a_min=0,
        )
        # 2. Compute those elements for which interest rate is != 0
        balance[~where_zero] = np.clip(
            (
                principal * ihat**num_periods
                - payment / interest_rate * (ihat**num_periods - 1)
            )[~where_zero],
            a_max=np.inf,
            a_min=0,
        )

    total_payments = np.clip(payment * num_periods, a_min=0, a_max=None)
    total_principal_payments = principal - balance
    total_interest_payments = total_payments - total_principal_payments

    return balance, total_interest_payments, total_payments


def _simple_compounding(
    principal: float,
    interest_rate_pct: float,
    payment: float,
    num_periods: int,
) -> tuple[float, float, float]:
    balance: float = principal
    total_interest_payments = 0.0
    total_payments = 0.0
    fract_interest_rate = interest_rate_pct / 100
    for _period in range(num_periods):
        this_interest = balance * fract_interest_rate
        total_interest_payments += this_interest
        balance += this_interest - payment
        total_payments += payment

    return balance, total_interest_payments, total_payments


def test_compounding():
    """Unit test comparing of manually-coded `compounding` and
    `simple_compounding` functions."""

    for principal in np.linspace(0, 20, 5):
        for interest_rate_pct in np.linspace(0, 20, 5):
            for payment in np.linspace(0, 2, 3):
                for num_periods in np.arange(0, 41, 5):
                    kw = {
                        "principal": principal,
                        "interest_rate_pct": interest_rate_pct,
                        "payment": payment,
                        "num_periods": num_periods,
                    }
                    sc_bal, sc_int, sc_pay = _simple_compounding(**kw)
                    c_bal, c_int, c_pay = compounding(**kw)
                    if not np.isclose(c_bal, sc_bal) and sc_bal >= 0 and principal != 0:
                        print(f"{kw=}, {c_bal=}, {sc_bal=}")
                    if not np.isclose(c_int, sc_int) and sc_int >= 0 and principal != 0:
                        print(f"{kw=}, {c_int=}, {sc_int=}")
                    if not np.isclose(c_pay, sc_pay) and sc_pay >= 0 and principal != 0:
                        print(f"{kw=}, {c_pay=}, {sc_pay=}")


def annual_to_monthly_rate_pct(
    annual_rate_pct: float | npt.NDArray,
) -> float | npt.NDArray:
    """Convert an annual percentage rate (APR) to equivalent rate if compounded monthly.

    Parameters
    ----------
    annual_rate_pct
        annual rate in percent (i.e., between 0 and 100)

    Returns
    -------
    monthly_rate_pct
        monthly rate in percent (i.e., between 0 and 100)

    """

    return 100 * ((1 + annual_rate_pct / 100) ** (1 / 12) - 1)


def calc_payment_for_payoff(
    principal: float | npt.NDArray,
    period_rate_pct: float | npt.NDArray,
    num_periods: float | npt.NDArray,
) -> float | npt.NDArray:
    """Calculate the payment necessary per period to pay off a loan with
    starting balance of `principal` and rate compounded once per priod
    `period_rate_pct` in `num_periods`.

    Parameters
    ----------
    principal
    period_rate_pct
    num_periods

    Returns
    -------
    period_payment

    """

    period_rate = period_rate_pct / 100
    compounded_rate = (1 + period_rate) ** num_periods
    period_payment = period_rate * principal * compounded_rate / (compounded_rate - 1)

    return period_payment


def details_and_pie_charts(info: Info, disp: bool, plot: bool):
    """Print out and optionally plot pie charts for details (dictionaries)
    within `info`.

    Parameters
    ----------
    info
        dict as returned by `generate_report` function
    disp
        whether to print text information to display
    plot
        whether (True) or not (False) to make pie charts

    """

    for k, v in info.items():
        if not isinstance(v, dict):
            continue

        title = k.replace("_", " ")
        if disp:
            make_headline(title)

        numerical_values = {}
        for k, v in sorted(
            v.items(),
            key=lambda x: x[1] if isinstance(x[1], (float, int)) else -1,
            reverse=True,
        ):
            t = k.replace("_", " ")
            if isinstance(v, float):
                valstr = f"{v:10,.3f}"
                numerical_values[t] = v
            elif isinstance(v, int):
                valstr = f"{v:10,d}"
                numerical_values[t] = v
            elif isinstance(v, str):
                valstr = v
            else:
                raise TypeError(f"{type(v) = }")

            if disp:
                print((t + " ").ljust(43, "-") + valstr)

        total = sum(numerical_values.values())

        if disp:
            print(" ".ljust(43) + "-" * 10)
            print("Total ".ljust(43) + f"{total:10,.3f}")

        if plot:
            fig, ax = plt.subplots(figsize=(12, 12), dpi=80)
            ax.pie(
                x=list(numerical_values.values()),
                labels=[f"{k} (${v:,.1f}k)" for k, v in numerical_values.items()],
                wedgeprops={"edgecolor": "white", "linewidth": 1},
            )
            ax.set_title(f"{title.title()} = ${sum(numerical_values.values()):,.1f}k")
            fig.tight_layout()


def print_info(info: Info, summary_only: bool = False, plot: bool = False):
    """Print summary and possibly details; optionally make pie charts for
    details.

    Parameters
    ----------
    info
    summary_only
    plot

    """

    globals().update(info)
    # pylint: disable=undefined-variable
    print(f"Purchase price:              {purchase_price:7,.1f}   k$")
    print(f"Down payment pct:            {down_payment/purchase_price * 100:7,.1f}   %")
    print(f"Down payment:                {down_payment:7,.1f}   k$")
    print(f"Principal:                   {principal:5,.0f}     k$")
    print(f"Loan term:                   {loan_period_years:5,.0f}     years")
    print(f"Sold after:                  {home_held_years:5,.0f}     years")
    print(f"Points:                      {points:5,.0f}")
    print(f"Points fee:                  {point_fee:7,.1f}   k$")
    print(f"Pct reduction due to points: {point_pct:8,.2f}  %")
    print(f"Resulting APR:               {apr:9.3f} %", end="")
    print(f" (monthly interest = {monthly_rate_pct:.5f}%)")
    print(f"Monthly mortgage payment:    {monthly_mortgage_payment:9,.3f} k$")
    print(
        f"Total monthly cost:          {total_monthly_cost :9,.3f} k$"
        " (incl. maint. expenses)"
    )
    print(
        f"Lifetime interest paid:      {total_interest_paid:7,.1f}   k$"
        f" ({interest_per_month:,.3f} k$/mo)"
    )
    print(f"Lifetime mortgage payments:  {total_mortgage_payments:7,.1f}   k$")
    print(f"Home value at resale:        {home_value_at_resale:7,.1f}   k$")
    print(f"Loan balance at resale:      {loan_balance_at_resale:7,.1f}   k$")
    print(f"Home value owned at resale:  {total_value_owned_at_resale:7,.1f}   k$")
    print(f"Total cost of ownership:     {total_cost_of_ownership:7,.1f}   k$")
    print(f"Net at resale:               {net:7,.1f}   k$")
    print(f"Net profit or loss per month {net_per_month:+9,.3f} k$")
    print(f"Cash at closing:             {total_cash_at_closing:7,.1f}   k$")

    if not summary_only:
        details_and_pie_charts(info=info, disp=True, plot=plot)


def generate_report(  # pylint: disable=too-many-arguments, too-many-locals, too-many-statements
    *,
    purchase_price: float,
    down_payment_pct: float,
    points: float,
    loan_period_years: float,
    home_held_years: float,
    apr: float,
    property_tax: float,
    realtor_fee_pct: float,
    homeowners_insurance_monthly: float,
    mortgage_tax_pct: float,
    attorneys_fees: float,  # pylint: disable=unused-argument
    title_insurance_pct: float,
    underwriting_fee: float,  # pylint: disable=unused-argument
    credit_report_fee: float,  # pylint: disable=unused-argument
    application_fee: float,  # pylint: disable=unused-argument
    inspection_cost: float,  # pylint: disable=unused-argument
    cesspool_inspection_cost: float,
    appraisal_fee: float,  # pylint: disable=unused-argument
    survey_cost: float,  # pylint: disable=unused-argument
    recording_fee: float,  # pylint: disable=unused-argument
    transfer_tax_pct: float,
    courier_fee: float,  # pylint: disable=unused-argument
    lead_paint_inspection: float,  # pylint: disable=unused-argument
    pest_inspection: float,  # pylint: disable=unused-argument
    moving_costs: float,  # pylint: disable=unused-argument
    annual_cesspool_maintenance_cost: float,
    has_cesspool: bool,
    annual_inflation_pct: float,
    annual_maintenance_cost: float,
    disp: bool = True,
    summary_only: bool = False,
    plot: bool = True,
) -> Info:
    """Generate a full report for a home loan.

    Parameters
    ----------
    purchase_price
        purchase price of the house
    down_payment_pct
        down payment as a percent of the purchase price of the house
    points
        number of points purchased to reduce the loan rate; here, each point
        costs an additional 1% of the purchase price of the house in additional
        down payment and reduces the apr by 0.25%
    loan_period_years
        period of the loan in years; e.g., 5, 15, or 30
    home_held_years
        number of years the house is held before being re-sold (can be a
        part of a year, such as 5.75)
    apr
        "annualized percentage rate," which is defined here as the rate that
        will be compounded each month multiplied by twelve
    property_tax
        annual property tax
    realtor_fee_pct
        percentage that realtor will take on TOP of purchase price; if realtor
        fee is rolled into the purchase price, specify 0 here
    homeowners_insurance_monthly
        insurance paid monthly to protect the house; also include flood, fire,
        or other disaster insurance in this number
    mortgage_tax_pct
        mortgage recording tax (only imposed by a few states: AL, FL, KS, MN,
        NY, OK, and TN at the time of writing)
    attorneys_fees
        fees charged by real estate attorney to review contracts (etc.)
    title_insurance_pct
        cost of insurance on title during tranfer
    underwriting_fee
        fee imposed by bank for underwriting the loan
    credit_report_fee
        fee imposed by bank for generating a credit report
    application_fee
        fee imposed by bank for loan application
    inspection_cost
        cost for property to be inspected
    cesspool_inspection_cost
        cost to have a cesspool inspected; forced to 0 if `has_cesspool` is
        False
    appraisal_fee
        cost for having property appraised (required by bank for loan)
    survey_cost
        cost to have property surveyed
    recording_fee
        government fee for recording the title (closing cost)
    transfer_tax_pct
        tax state imposes on transferring a title (closing cost)
    courier_fee
        fee for courier service for closing (closing cost)
    lead_paint_inspection
        cost for getting lead paint inspection
    pest_inspection
        cost for getting pest inspection
    moving_costs
        cost to move into the new home
    has_cesspool
        whether property has a cesspool
    annual_cesspool_maintenance_cost
        cost per year to maintain the cesspool; this is forced to be 0 if
        `has_cesspool` is False
    annual_inflation_pct
        percent inflation expected annually
    annual_maintenance_cost
        maintenance cost per year in 1st-year (non-inflated) dollars
    disp
        allow printing out summary and, optionally, detail info
    summary_only
        do not print details, only print the high-level summary info
    plot
        whether to plot pie charts showing components of computed totals

    Returns
    -------
    info

    """

    # pylint: disable=possibly-unused-variable

    if has_cesspool:
        cesspool_maintenance_cost_monthly = (
            cesspool_inspection_cost * 18 / 24 + annual_cesspool_maintenance_cost / 12
        )
    else:
        if cesspool_inspection_cost != 0:
            #logger.warning("Forcing cesspool inspection cost to 0 since no cesspool")
            cesspool_inspection_cost = 0
        if annual_maintenance_cost != 0:
            #logger.warning("Forcing cesspool maintenance cost to 0 since no cesspool")
            annual_cesspool_maintenance_cost = 0
        cesspool_maintenance_cost_monthly = 0

    loan_period_months = loan_period_years * 12
    home_held_months = home_held_years * 12
    # NOTE: assuming we're subtracting 4% Realtor fees at resale of home
    home_value_at_resale = (
        purchase_price * (1 + annual_inflation_pct / 100) ** home_held_years
    ) * 0.96

    annual_maintenance_cost_pct = 100 * annual_maintenance_cost / purchase_price

    monthly_maintenance_cost = annual_maintenance_cost / 12

    total_maintenance_cost = (
        purchase_price
        * (annual_maintenance_cost_pct / 100)
        * sum(
            (1 + annual_inflation_pct / 100) ** i for i in range(int(home_held_years))
        )
    )
    amortized_annual_maintenance_cost = total_maintenance_cost / home_held_years
    amortized_monthly_maintenance_cost = amortized_annual_maintenance_cost / 12

    point_fee = 0.0
    point_pct = 0.0
    if points:
        point_fee = (points / 100) * purchase_price
        point_pct = points / 4
        apr -= point_pct

    # == Print header == #

    if disp:
        make_headline("Headlines")

    # == Loan stuff

    principal = purchase_price * (1 - down_payment_pct / 100)  # k$
    down_payment = purchase_price * down_payment_pct / 100

    monthly_rate_pct = apr / 12
    monthly_mortgage_payment = calc_payment_for_payoff(
        principal=principal,
        period_rate_pct=monthly_rate_pct,
        num_periods=loan_period_months,
    )

    # ==

    property_tax_monthly = property_tax / 12

    labels_non_mortgage_monthly_costs = [
        "property_tax_monthly",
        "homeowners_insurance_monthly",
        "monthly_maintenance_cost",
    ]
    if has_cesspool:
        labels_non_mortgage_monthly_costs.append("cesspool_maintenance_cost_monthly")

    lcls = locals()
    non_mortgage_monthly_costs = {
        lbl: lcls[lbl] for lbl in labels_non_mortgage_monthly_costs
    }
    total_non_mortgage_monthly_costs = sum(non_mortgage_monthly_costs.values())

    # ==

    labels_monthly_cost = [
        "monthly_mortgage_payment"
    ] + labels_non_mortgage_monthly_costs
    lcls = locals()
    monthly_cost = {lbl: lcls[lbl] for lbl in labels_monthly_cost}

    total_monthly_cost = sum(monthly_cost.values())

    (
        loan_balance_at_resale,
        total_interest_paid,
        total_mortgage_payments,
    ) = compounding(
        principal=principal,
        interest_rate_pct=monthly_rate_pct,
        payment=monthly_mortgage_payment,
        num_periods=min(home_held_months, loan_period_months),
    )
    total_principal_paid = total_mortgage_payments - total_interest_paid
    total_value_owned_at_resale = home_value_at_resale - loan_balance_at_resale
    interest_per_month = total_interest_paid / home_held_months

    loan_lifetime_non_mortgage_monthly_costs = (
        total_non_mortgage_monthly_costs * home_held_months
    )

    # ==

    mortgage_tax = principal * mortgage_tax_pct / 100
    title_insurance = purchase_price * title_insurance_pct / 100
    realtor_fee = purchase_price * realtor_fee_pct / 100
    transfer_tax = purchase_price * transfer_tax_pct / 100

    labels_one_time_costs = [
        "realtor_fee",
        "point_fee",
        "mortgage_tax",
        "title_insurance",
        "attorneys_fees",
        "underwriting_fee",
        "credit_report_fee",
        "application_fee",
        "inspection_cost",
        "appraisal_fee",
        "survey_cost",
        "recording_fee",
        "transfer_tax",
        "courier_fee",
        "lead_paint_inspection",
        "pest_inspection",
        "moving_costs",
    ]
    if has_cesspool:
        labels_one_time_costs.append("cesspool_inspection_cost")

    lcls = locals()
    one_time_costs = {lbl: lcls[lbl] for lbl in labels_one_time_costs}

    total_one_time_costs = sum(one_time_costs.values())

    labels_cost_of_ownership = [
        "down_payment",
        "total_principal_paid",
        "total_interest_paid",
        "total_one_time_costs",
        "loan_lifetime_non_mortgage_monthly_costs",
    ]
    lcls = locals()
    cost_of_ownership = {lbl: lcls[lbl] for lbl in labels_cost_of_ownership}
    total_cost_of_ownership = sum(cost_of_ownership.values())

    net = home_value_at_resale - (loan_balance_at_resale + total_cost_of_ownership)
    net_per_month = net / home_held_months

    # ==

    taxes_at_closing = property_tax / 2
    monthly_payment_at_closing = 0 * monthly_mortgage_payment
    homeowners_insurance_at_closing = 2 * homeowners_insurance_monthly

    # labels_non_down_payment_cash_at_closing = [
    #     "taxes_at_closing",
    #     "monthly_payment_at_closing",
    #     "homeowners_insurance_at_closing",
    #     "total_one_time_costs",
    # ]
    # lcls = locals()
    # non_down_payment_cash_at_closing = {
    #     lbl: lcls[lbl] for lbl in labels_non_down_payment_cash_at_closing
    # }
    # total_non_down_payment_cash_at_closing = sum(
    #     non_down_payment_cash_at_closing.values()
    # )

    labels_cash_at_closing = [
        "down_payment",
        "taxes_at_closing",
        "monthly_payment_at_closing",
        "homeowners_insurance_at_closing",
        "total_one_time_costs",
    ]
    lcls = locals()
    cash_at_closing = {lbl: lcls[lbl] for lbl in labels_cash_at_closing}
    total_cash_at_closing = sum(cash_at_closing.values())

    # ==

    info: Info = copy(locals())
    info.pop("lcls")
    info.pop("disp")
    info.pop("plot")
    info.pop("summary_only")

    if disp:
        print_info(info, summary_only=summary_only, plot=plot)

    return info


def merge_params(
    default_params: Params | None = None, **updated_params
) -> Params:
    """Create a new set of params by merging `updated_params` into
    `default_params` without modifying either of the original objects.

    Parameters
    ----------
    default_params
    **updated_params

    Returns
    -------
    merged_params

    """

    merged_params = dict(default_params) if default_params else {}

    if updated_params:
        merged_params.update(updated_params)

    return merged_params


def extract_params_from_info(info: Info) -> Params:
    """Extract a params dict from the relevant key/value pairs in `info`.

    Parameters
    ----------
    info

    Returns
    -------
    params

    """

    sig = signature(generate_report)
    return {k: info[k] for k in sig.parameters.keys() if k in info}


def make_dataframe(infos: Info | Sequence[Info]) -> pd.DataFrame:
    """Turn an `info` dict or a sequence thereof into a Pandas DataFrame.

    Note that dict or list values in `info` are ignored; only "scalar" values
    (including strings, ints, floats, etc.) will be used, while lists and dicts
    within `info` will not be included (and their keys will not show up as
    columns in the resulting DataFrame).

    Parameters
    ----------
    infos
        one `info` dict or a seqence thereof

    Returns
    -------
    info_df
        DataFrame containing all scalar (string, int, float) fields in the
        `info` dicts

    """

    if isinstance(infos, dict):
        infos_ = [infos]
    elif isinstance(infos, Sequence):
        infos_ = infos
    else:
        raise TypeError

    new_infos = []
    for info in infos_:
        this_info = {}
        for k, v in info.items():
            if isinstance(v, (list, dict)):
                continue
            this_info[k] = v
        new_infos.append(this_info)
    info_df = pd.DataFrame(new_infos)

    return info_df


def evaluate(
    x: float, vary_name: str, target_name: str, params: Params
) -> tuple[float, Info]:
    """Evaluate target param value as a function of a value evaluated for
    another param.

    Parameters
    ----------
    x
        value of independent variable
    vary_name
        independent variable name
    target_name
        name of dependent variable
    params
        values to use for all kwargs required by `generate_report`

    Returns
    -------
    y
        value of `target_name` when parameter `vary_name` is given value `x`
    info
        info dict as returned by `generate_report`

    """

    params_d = dict(params)
    params_d[vary_name] = x
    info = generate_report(disp=False, plot=False, **params_d)
    return info[target_name], info


def quadratic_cost(
    x: float,
    vary_name: str,
    target_name: str,
    target_val: float,
    params: Params,
) -> float:
    """Cost function.

    Parameters
    ----------
    x
    vary_name
    target_name
    target_val
    params

    Returns
    -------
    cost

    """

    y, _info = evaluate(
        x=x, vary_name=vary_name, target_name=target_name, params=params
    )
    return (y - target_val) ** 2


def generic_optimizer(  # pylint: disable=too-many-arguments
    *,
    vary_name: str,
    bounds: Optional[Sequence[float]],
    target_name: str,
    target_val: float,
    default_params: Params | None = None,
    **updated_params
) -> tuple[float, float, Info]:
    """Vary `vary_name` within optional `bounds` such that target `target_name`
    reaches `target_val` given `default_params` with any values specified in
    `updated_params` updated accordingly.

    Parameters
    ----------
    vary_name
    bounds
    target_name
    target_val
    default_params
    **updated_params

    Returns
    -------
    x
        value of `vary_name` that optimizes `target_name` to be close to `target_val`
    y
        optimized value of 'target_name` (if optimization succeeded, should be
        within epsilon of `target_val`)
    info
        info dict as returned by `generate_report`

    """

    params = merge_params(default_params=default_params, **updated_params)

    result = optimize.minimize_scalar(
        fun=quadratic_cost,
        bounds=bounds,
        args=(vary_name, target_name, target_val, params),
        options={"disp": False},
    )
    x = result.x
    y, info = evaluate(x=x, vary_name=vary_name, target_name=target_name, params=params)

    return x, y, info


def brute_force_optimizer(  # pylint: disable=too-many-arguments
    *,
    vary_name: str,
    values: Sequence[float],
    target_name: str,
    target_val: float,
    default_params: Params | None = None,
    **updated_params
) -> tuple[float, float, Info, npt.NDArray, npt.NDArray]:
    """Try every value in `values` for param `vary_name` and see which one
    yields a param `target_name` nearest to `target_val`. Start with
    `default_params` updated with `updated_params` before modifying
    `vary_name`.

    Parameters
    ----------
    vary_name
    values
    target_name
    target_val
    default_params
    **updated_params

    Returns
    -------
    x
    y
    info
    ys
    costs

    """

    params = merge_params(default_params=default_params, **updated_params)

    costs = np.empty(shape=(len(values),))
    ys = np.empty_like(costs)
    infos = []
    for i, x in enumerate(values):
        y, info = evaluate(
            x=x, vary_name=vary_name, target_name=target_name, params=params
        )
        ys[i] = y
        infos.append(info)
        costs[i] = (y - target_val) ** 2
    opt_idx = np.argmin(costs)
    x = values[opt_idx]
    y = ys[opt_idx]
    info = infos[opt_idx]

    return x, y, info, ys, costs


def calc_down_payment_from_cash_at_closing(
    total_cash_at_closing: float,
    min_down_payment_pct: float,
    max_down_payment_pct: float,
    default_params: Params | None = None,
    **updated_params
) -> tuple[float, float, Info]:
    """Calculate the down payment possible given total cash available at
    closing.

    Parameters
    ----------
    total_cash_at_closing
    min_down_payment_pct
    max_down_payment_pct
    default_params
    **updated_params

    Returns
    -------
    down_payment
    down_payment_pct
    info

    """

    assert 0 <= min_down_payment_pct <= 100
    assert 0 <= max_down_payment_pct <= 100

    down_payment_pct, total_cash_at_closing, info = generic_optimizer(
        vary_name="down_payment_pct",
        bounds=[min_down_payment_pct, max_down_payment_pct],
        target_name="total_cash_at_closing",
        target_val=total_cash_at_closing,
        default_params=default_params,
        **updated_params
    )

    return info["down_payment"], down_payment_pct, info


def calc_down_payment_for_monthly_cost(
    total_monthly_cost: float,
    min_down_payment_pct: float,
    max_down_payment_pct: float,
    default_params: Params | None = None,
    **updated_params
) -> tuple[float, float, dict[str, Any]]:
    """Calculate the down payment required to achieve a target monthly payment
    on the house (including homeowners insurance, property taxes, and mortgage
    payemnts).

    Parameters
    ----------
    total_monthly_cost
    min_down_payment_pct
    max_down_payment_pct
    default_params
    **updated_params

    Returns
    -------
    down_payment
    down_payment_pct
    info

    """

    assert 0 <= min_down_payment_pct <= 100
    assert 0 <= max_down_payment_pct <= 100

    down_payment_pct, total_monthly_cost, info = generic_optimizer(
        vary_name="down_payment_pct",
        bounds=[min_down_payment_pct, max_down_payment_pct],
        target_name="total_monthly_cost",
        target_val=total_monthly_cost,
        default_params=default_params,
        **updated_params
    )

    return info["down_payment"], down_payment_pct, info
