defmodule Exgrad.MLP do
  alias Exgrad.{Expr, Layer}

  @type t :: {:mlp, Expr.t()}

  def from_config(definition, opts) do
    seed = Keyword.get(opts, :seed, 0)
    :rand.seed(:default, seed)
    from_config(definition)
  end

  def from_config([nin | _] = definition) do
    xs =
      if is_list(nin) do
        nin
      else
        for _ <- 1..nin, do: 0
      end
      |> Enum.with_index(&Expr.value(&1, "x:#{&2}"))

    {:mlp, from_config(definition, xs, 0)}
  end

  defp from_config([_def_1, def_2], xs, n) do
    build_layer(xs, def_2, :linear, n) |> unpack()
  end

  defp from_config([_def_1, def_2 | rest], xs, n) do
    from_config([def_2 | rest], build_layer(xs, def_2, :relu, n), n + 1)
  end

  defp build_layer(xs, def, afun, n) do
    {nout, activation} = get_layer_def(def, afun)
    {:layer, exprs} = Layer.from_config(xs, nout, activation: activation, label: to_string(n))
    exprs
  end

  def from_parameters(parameters) do
    {_, ws, _} = hd(parameters)
    nin = ws |> hd() |> length()
    xs =
      1..nin
      |> Enum.map(fn _ -> 0 end)
      |> Enum.with_index(&Expr.value(&1, "x:#{&2}"))

    {:mlp, from_parameters(parameters, xs, 0) |> unpack()}
  end

  def from_parameters([params], xs, n) do
    {:layer, layer} = Layer.from_parameters(params, xs, label: to_string(n))
    layer
  end

  def from_parameters([params | rest], xs, n) do
    {:layer, layer} = Layer.from_parameters(params, xs, label: to_string(n))
    from_parameters(rest, layer, n + 1)
  end

  def to_parameters(model, opts) do
    map = to_parameters_map(model)
    definition = Keyword.fetch!(opts, :definition)
    to_parameters(map, definition, 0)
  end

  defp to_parameters(map, [def_1, def_2], n) do
    nin = get_layer_nin(def_1)
    {nout, activation} = get_layer_def(def_2, :linear)

    [Layer.to_parameters(map, nin, nout, activation: activation, label: n)]
  end

  defp to_parameters(map, [def_1, def_2 | rest], n) do
    nin = get_layer_nin(def_1)
    {nout, activation} = get_layer_def(def_2, :relu)

    layer = Layer.to_parameters(map, nin, nout, label: n, activation: activation)
    [layer | to_parameters(map, [def_2 | rest], n + 1)]
  end

  defp get_layer_nin({nin, _afun}), do: nin
  defp get_layer_nin(nin), do: nin

  defp get_layer_def({nout, afun}, _def_afun), do: {nout, afun}
  defp get_layer_def(nout, def_afun), do: {nout, def_afun}

  defp unpack([x]), do: x
  defp unpack(x), do: x

  defp to_parameters_map(expr) do
    Expr.reduce(expr, %{}, fn
      %Expr{label: "w:" <> _rest} = n, acc ->
        Map.put_new(acc, n.label, n.value)

      %Expr{label: "b:" <> _rest} = n, acc ->
        Map.put_new(acc, n.label, n.value)

      _, acc ->
        acc
    end)
  end
end
