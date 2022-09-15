#
# Class for current-driven ODE for interface utilisation
#
import pybamm
from .base_utilisation import BaseModel


class CurrentDriven(BaseModel):
    """Current-driven ODE for interface utilisation


    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'negative' or 'positive'
    options : dict, optional
        A dictionary of options to be passed to the model.
    reaction_loc : str
        Where the reaction happens: "x-average" (SPM, SPMe, etc),
        "full electrode" (full DFN), or "interface" (half-cell model)

    **Extends:** :class:`pybamm.interface utilisation.BaseModel`
    """

    def __init__(self, param, domain, options, reaction_loc):
        super().__init__(param, domain, options)
        self.reaction_loc = reaction_loc

    def get_fundamental_variables(self):
        domain = self.domain
        Domain = domain.capitalize()

        if self.reaction_loc == "full electrode":
            u = pybamm.Variable(
                f"{Domain} electrode interface utilisation variable",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
        elif self.reaction_loc == "x-average":
            u_xav = pybamm.Variable(
                f"X-averaged {domain} electrode interface utilisation variable",
                domain="current collector",
            )
            u = pybamm.PrimaryBroadcast(u_xav, f"{domain} electrode")
        else:
            u = pybamm.Variable(
                "Lithium metal interface utilisation variable",
                domain="current collector",
            )
        variables = self._get_standard_interface_utilisation_variables(u)

        return variables

    def set_rhs(self, variables):
        domain = self.domain
        Domain = domain.capitalize()

        if self.reaction_loc == "full electrode":
            u = variables[f"{Domain} electrode interface utilisation variable"]
            a = variables[f"{Domain} electrode surface area to volume ratio"]
            j = variables[f"{Domain} electrode interfacial current density"]
        elif self.reaction_loc == "x-average":
            u = variables[
                f"X-averaged {domain} electrode interface utilisation variable"
            ]
            a = variables[f"X-averaged {domain} electrode surface area to volume ratio"]
            j = variables[f"X-averaged {domain} electrode interfacial current density"]
        else:
            u = variables["Lithium metal interface utilisation variable"]
            a = 1
            j = variables["Lithium metal total interfacial current density"]

        beta = self.domain_param.beta_utilisation

        self.rhs = {u: beta * a * u * j}

    def set_initial_conditions(self, variables):
        domain = self.domain
        Domain = domain.capitalize()

        if self.reaction_loc == "full electrode":
            u = variables[f"{Domain} electrode interface utilisation variable"]
        elif self.reaction_loc == "x-average":
            u = variables[
                f"X-averaged {domain} electrode interface utilisation variable"
            ]
        else:
            u = variables["Lithium metal interface utilisation variable"]

        u_init = self.domain_param.u_init

        self.initial_conditions = {u: u_init}

    def set_events(self, variables):
        domain = self.domain
        Domain = domain.capitalize()

        if self.reaction_loc == "full electrode":
            u = variables[f"{Domain} electrode interface utilisation"]
        elif self.reaction_loc == "x-average":
            u = variables[f"X-averaged {domain} electrode interface utilisation"]
        else:
            u = variables["Lithium metal interface utilisation"]
        self.events.append(
            pybamm.Event(
                f"Zero {domain} electrode utilisation cut-off",
                pybamm.min(u),
                pybamm.EventType.TERMINATION,
            )
        )
