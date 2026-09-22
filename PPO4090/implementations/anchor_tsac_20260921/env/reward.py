"""可望远镜求和的唯一主奖励。"""
def delta_reward(before_satisfaction, after_satisfaction, acted_index, demand_mask, scale=100.0):
    n = int(demand_mask.sum())
    delta = (after_satisfaction-before_satisfaction)[demand_mask].sum()/n
    current_gain = (after_satisfaction[acted_index]-before_satisfaction[acted_index])/n
    historical_loss = current_gain-delta
    return float(scale*delta), {"current_beam_utility_gain": float(current_gain), "historical_utility_loss": float(historical_loss),
                               "delta_U": float(delta), "reward_scale": float(scale)}

