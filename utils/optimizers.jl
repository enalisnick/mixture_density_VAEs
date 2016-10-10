
# AdaM: https://arxiv.org/pdf/1412.6980v8.pdf
function adamUpdate(params, grads, adamParams, b1=.95, b2=.999, e=1e-8)
  adamParams["t"] += 1

  for paramType in ["w_encoder", "b_encoder", "w_decoder", "b_decoder"]
    if contains(paramType,"encoder") idx_limit = 3 else idx_limit = 2 end
    for param_idx in 1:idx_limit
      # mean term
      adamParams["m"][paramType][param_idx] = b1*adamParams["m"][paramType][param_idx] + (1-b1)*grads[paramType][param_idx]
      m_hat = adamParams["m"][paramType][param_idx] ./ (1-b1.^adamParams["t"])

      # variance term
      adamParams["v"][paramType][param_idx] = b2*adamParams["v"][paramType][param_idx] + (1-b2)*grads[paramType][param_idx].^2
      v_hat = adamParams["v"][paramType][param_idx] ./ (1-b2.^adamParams["t"])

      # update model param
      params[paramType][param_idx] -= adamParams["lr"] * m_hat./(sqrt(v_hat) + e)
    end
  end

  return params
end

