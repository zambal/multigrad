defmodule Nxgrad.MLP do
  alias Nxgrad.Layer

  @type t :: [Layer.t()]

  def from_config(definition, opts) do
    seed = Keyword.get(opts, :seed, 0)
    :rand.seed(:default, seed)
    from_config(definition)
  end

  def from_config([d1, d2]) do
    [build_layer(d1, d2, :linear)]
  end

  def from_config([d1, d2 | rest]) do
    [build_layer(d1, d2, :relu) | from_config([d2 | rest])]
  end

  def from_parameters(params) do
    Enum.map(params, &Layer.from_parameters/1)
  end

  def to_parameters(mlp) do
    Enum.map(mlp, &Layer.to_parameters/1)
  end

  defp build_layer(nin, nout, activation) do
    {nin, _} = get_layer_def(nin, nil)
    {nout, activation} = get_layer_def(nout, activation)
    Layer.from_config(nin, nout, activation: activation)
  end

  defp get_layer_def({n, afun}, _def_afun), do: {n, afun}
  defp get_layer_def(n, def_afun), do: {n, def_afun}
end
