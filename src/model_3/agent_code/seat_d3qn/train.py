import os


_variant = os.environ.get("SEAT_D3QN_VARIANT", "m2_3")
if _variant == "m2_3":
    from .variants.m2_3.train import end_of_round, game_events_occurred, setup_training
elif _variant == "m2_4":
    from .variants.m2_4.train import end_of_round, game_events_occurred, setup_training
elif _variant == "m2_4_c3":
    raise ValueError("The legacy C3 branch is archived. Use m2_4_c3_v2 for the reported League policy.")
elif _variant == "m2_4_c3_v2":
    from .variants.m2_4.c3_v2.train import (
        end_of_round,
        game_events_occurred,
        setup_training,
    )
elif _variant == "m2_5":
    from .variants.m2_5.train import (
        end_of_round,
        game_events_occurred,
        setup_training,
    )
elif _variant == "m2_0":
    from .variants.m2_0.train import end_of_round, game_events_occurred, setup_training
elif _variant == "m2_1":
    from .variants.m2_1.train import end_of_round, game_events_occurred, setup_training
elif _variant == "m2_2":
    from .variants.m2_2.train import end_of_round, game_events_occurred, setup_training
elif _variant == "m2_2_control":
    from .variants.m2_2_control.train import (
        end_of_round,
        game_events_occurred,
        setup_training,
    )
else:
    raise ValueError("Unknown SEAT_D3QN_VARIANT: " + repr(_variant))

__all__ = ("setup_training", "game_events_occurred", "end_of_round")
