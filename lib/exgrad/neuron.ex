defmodule Exgrad.Neuron do
  use Exgrad.Expr

  @type t :: {:neuron, Expr.t()}

  defe from_config(xs, opts \\ []) do
    nin = length(xs)
    label = get_label(opts)
    b = value(0, "b:#{label}")
    ws = Enum.map(0..Kernel.-(nin, 1), &(rand_number(nin) |> value("w:#{&1}:#{label}")))

    {:neuron, b + sum(map(ws, xs, &(&1 * &2)))}
  end

  defe from_parameters(b, ws, xs, opts \\ []) do
    label = get_label(opts)
    b = value(b, "b:#{label}")
    ws = Enum.with_index(ws, &value(&1, "w:#{&2}:#{label}"))

    {:neuron, b + sum(map(ws, xs, &(&1 * &2)))}
  end

  def to_parameters(parameter_map, nin, opts) do
    label = Keyword.fetch!(opts, :label)
    b = Map.fetch!(parameter_map, "b:#{label}")

    ws =
      for n <- 0..(nin - 1) do
        Map.fetch!(parameter_map, "w:#{n}:#{label}")
      end

    {b, ws}
  end

  def get_label(opts) do
    Keyword.get_lazy(opts, :label, fn ->
      System.unique_integer([:positive]) |> to_string()
    end)
  end

  defp rand_number(nin), do: (:rand.uniform() * 2 - 1) * :math.sqrt(2.0 / nin)
end
