defmodule Axongrad.MLP do
  alias Axongrad.{Activation, Layer}

  def from_config([nin | nouts], opts) do
    seed = Keyword.get(opts, :seed, 0)
    :rand.seed(:default, seed)

    lin = Layer.from_config(nin)
    from_config(nouts, {lin, %{}}, 0)
  end

  def from_config([d1, d2], model, n) do
    build_layer(model, d1, d2, :linear, n)
  end

  def from_config([d1, d2 | rest], model, n) do
    from_config([d2 | rest], build_layer(model, d1, d2, :relu, n), n + 1)
  end

  def from_parameters(params) do
    nin = nin_from_params(params)
    lin = Layer.from_config(nin)
    from_parameters(params, {lin, %{}}, 0)
  end

  def from_parameters([], {model, params}, _n) do
    %Axon{nodes: %{"xs" => {nil, nin}}} = model
    {init_fn, predict_fn} = Axon.build(model)
    init_fn.()
  end

  def from_parameters([d1, d2 | rest], model, n) do
    from_parameters([d2 | rest], build_layer(model, d1, d2, :relu, n), n + 1)
  end

  def to_parameters(mlp) do
    Enum.map(mlp, &Layer.to_parameters/1)
  end

  defp build_layer(model, nin, nout, activation, n) do
    {nin, _} = get_layer_def(nin, nil)
    {nout, activation} = get_layer_def(nout, activation)
    Layer.from_config(model, nin, nout, activation: activation, label: "dense_#{n}")
  end

  defp parameterized_layer(model, layer, n) do
    Layer.from_parameters(model, layer, label: "dense_#{n}")
  end


  defp get_layer_def({n, afun}, _def_afun), do: {n, afun}
  defp get_layer_def(n, def_afun), do: {n, def_afun}

  defp nin_from_params([{_b, [w | _] = _ws, _act} | _rest]) do
    length(w)
  end
end
