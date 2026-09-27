

ACTIONS              = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_STRING_TO_ID  = {action: id for id, action in enumerate(ACTIONS)}
N_ACTIONS            = len(ACTIONS)
N_STATES             = 32768