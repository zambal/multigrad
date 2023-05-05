defmodule Exgrad.Layer do
  alias Exgrad.{Activation, Expr, Neuron}

  @type t :: {:layer, [Expr.t()]}

  def from_config(xs, nout, opts) do
    label = Neuron.get_label(opts)
    activation = Keyword.get(opts, :activation)

    neurons =
      for n <- 0..(nout - 1) do
        {:neuron, n} = Neuron.from_config(xs, label: "#{n}_#{label}")
        if activation == :linear, do: n, else: Activation.afun(:relu).(n)
      end

    {:layer, neurons}
  end

  def from_parameters({b, ws, activation}, xs, opts) do
    label = Neuron.get_label(opts)

    neurons =
      Stream.zip(b, ws) |> Enum.with_index(fn {b, ws}, n ->
        {:neuron, n} = Neuron.from_parameters(b, ws, xs, label: "#{n}_#{label}")
        if activation == :linear, do: n, else: Activation.afun(:relu).(n)
      end)

    {:layer, neurons}
  end

  def to_parameters(parameter_map, nin, nout, opts) do
    label = Keyword.fetch!(opts, :label)

    {bs, ws} =
      Enum.reduce(0..(nout - 1), {[], []}, fn n, {bs, ws} ->
        {b, w} = Neuron.to_parameters(parameter_map, nin, label: "#{n}_#{label}")
        {[b | bs], [w | ws]}
      end)

    {Enum.reverse(bs), Enum.reverse(ws), Keyword.fetch!(opts, :activation)}
  end

  def transpose(matrix) do
    transpose(matrix, [], [], [])
  end

  defp transpose([[y | ys] | rest], xs, new_ys, acc) do
    transpose(rest, [ys | xs], [y | new_ys], acc)
  end

  defp transpose([], [[] | _], new_ys, acc) do
    Enum.reverse([Enum.reverse(new_ys) | acc])
  end

  defp transpose([], xs, new_ys, acc) do
    transpose(Enum.reverse(xs), [], [], [Enum.reverse(new_ys) | acc])
  end


  defp transpose2([ys | xs]) do
    transpose2(ys, xs)
  end

  defp transpose2([y | ys], xs) do

  end

end
