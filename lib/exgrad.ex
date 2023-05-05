defmodule Exgrad do
  @behaviour Multigrad

  use Exgrad.Expr
  alias Exgrad.MLP

  defmacro __using__(_opts) do
    quote do
      use Exgrad.Expr
      import Exgrad
    end
  end

  # API

  def build({:neuron, expr}), do: setup_loss(expr)
  def build({:layer, exprs}), do: setup_loss(exprs)
  def build({:mlp, exprs}), do: setup_loss(exprs)

  # Behaviour

  def to_training_data(data, batch_size) do
    data
    |> Stream.map(fn {x, y} ->
      label_map =
        x
        |> Stream.with_index()
        |> Enum.reduce(%{}, fn {x, n}, acc ->
          Map.put(acc, "x:#{n}", x * 1.0)
        end)

      y
      |> Stream.with_index()
      |> Enum.reduce(label_map, fn {y, n}, acc ->
        Map.put(acc, "yg:#{n}", y * 1.0)
      end)
    end)
    |> Stream.chunk_every(batch_size)
  end

  def from_config(definition, opts) do
    {:ok, MLP.from_config(definition, opts) |> build()}
  end

  def from_parameters(parameters) do
    {:ok, MLP.from_parameters(parameters) |> build()}
  end

  def to_parameters(model, opts) do
    MLP.to_parameters(model, opts)
  end

  def train({model, _loss}, samples, rate) do
    Enum.reduce(samples, {model, 0}, fn batch, {model, _loss} ->
      run_batch(model, batch, rate)
    end)
  end

  # Private

  defp run_batch(model, samples, rate) do
    # length(samples)
    length = 1

    {grads, loss} =
      Enum.reduce(samples, {%{}, 0}, fn values, {grads, loss} ->
        {model, grads} = run(model, values, grads)
        {grads, loss + get_loss_value(model)}
      end)

    model =
      Expr.forward(model, fn
        %Expr{label: "w:" <> _rest} = n ->
          grad = Map.get(grads, n.label, 0) / length
          Expr.put(n, n.value + rate * grad)

        %Expr{label: "b:" <> _rest} = n ->
          grad = Map.get(grads, n.label, 0) / length
          Expr.put(n, n.value + rate * grad)

        n ->
          n
      end)

    {model, loss}
  end

  defp run(expr, values, grads) do
    expr =
      Expr.forward(expr, fn
        %Expr{label: "x:" <> _rest} = n ->
          Expr.put(n, Map.fetch!(values, n.label))

        %Expr{label: "yg:" <> _rest} = n ->
          Expr.put(n, Map.fetch!(values, n.label))

        n ->
          n
      end)

    grads =
      Expr.backward_reduce(expr, grads, fn
        %Expr{label: "w:" <> _rest} = n, acc ->
          Map.update(acc, n.label, n.grad || 0, &(&1 + (n.grad || 0)))

        %Expr{label: "b:" <> _rest} = n, acc ->
          Map.update(acc, n.label, n.grad || 0, &(&1 + (n.grad || 0)))

        _, acc ->
          acc
      end)

    {expr, grads}
  end

  defe loss(prediction, goal), do: map(prediction, goal, &((&1 - &2) ** 2))

  defp setup_loss(exprs) when is_list(exprs) do
    {predictions, goals} = setup_ys(exprs)
    loss(predictions, goals) |> Expr.sum()
  end

  defp setup_loss(expr) do
    {[prediction], [goal]} = setup_ys([expr])
    loss(prediction, goal)
  end

  defp setup_ys(exprs) do
    l = length(exprs)
    predictions = Enum.with_index(exprs, fn expr, n -> Expr.value(expr, "yp:#{n}") end)
    goals = for n <- 0..(l - 1), do: Expr.value(0, "yg:#{n}")

    {predictions, goals}
  end

  defp get_loss_value(%Expr{} = expr), do: expr.value
end
