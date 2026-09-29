import os


_variant = os.environ.get("SEAT_D3QN_VARIANT", "m2_3")
if _variant == "m2_3":
    from .variants.m2_3.callbacks import act, setup
elif _variant == "m2_4":
    from .variants.m2_4.callbacks import act, setup
elif _variant == "m2_4_c3":
    raise ValueError("The legacy C3 branch is archived. Use m2_4_c3_v2 for the reported League policy.")
elif _variant == "m2_4_c3_v2":
    from .variants.m2_4.c3_v2.callbacks import act, setup
elif _variant == "m2_5":
    from .variants.m2_5.callbacks import act, setup
elif _variant == "m2_0":
    from .variants.m2_0.callbacks import act, setup
elif _variant == "m2_1":
    from .variants.m2_1.callbacks import act, setup
elif _variant == "m2_2":
    from .variants.m2_2.callbacks import act, setup
elif _variant == "m2_2_control":
    from .variants.m2_2_control.callbacks import act, setup
else:
    raise ValueError("Unknown SEAT_D3QN_VARIANT: " + repr(_variant))

__all__ = ("setup", "act")
