defmodule Exgrad.Expr do
  alias Exgrad.Expr

  defstruct [:value, :grad, :forward, :backward, :nodes, :label, :op]

  @type t :: %Expr{
          value: nil | number,
          grad: nil | number,
          forward: expr_fun,
          backward: expr_fun,
          nodes: [t],
          label: label,
          op: op
        }

  @type label :: String.Chars.t()
  @type expr_fun :: (t -> t)
  @type value :: number | t
  @type op :: :value | :+ | :- | :* | :/ | :** | :neg | :relu | :abs

  @type vector :: t | [t]

  defmacro __using__(_opts) do
    quote do
      alias Exgrad.Expr
      import Exgrad.Expr.Algebra, only: [expr: 1, run: 1, defe: 2]
    end
  end

  @spec value(value, label) :: t
  def value(value, label \\ nil)

  def value(value, label) when is_number(value) do
    %Expr{value: value, label: label, forward: & &1, backward: & &1, nodes: [], op: :value}
  end

  def value(%Expr{} = n, label) do
    if is_nil(label), do: n, else: %Expr{n | label: label}
  end

  @spec put(t, number) :: t
  def put(%Expr{op: :value} = n, value), do: %Expr{n | value: value}

  @spec put(t, number, label) :: t
  def put(%Expr{op: :value} = n, value, label), do: %Expr{n | value: value, label: label}

  @spec forward(vector, expr_fun) :: vector
  def forward(%Expr{nodes: ns} = n, fun) do
    n = %Expr{n | nodes: forward(ns, fun)}
    fun.(n.forward.(n))
  end

  def forward([n | ns], fun) do
    [forward(n, fun) | forward(ns, fun)]
  end

  def forward([], _fun), do: []

  @spec forward_reduce(vector, any, (t, any -> {t, any})) :: {vector, any}
  def forward_reduce(%Expr{nodes: ns} = n, acc, fun) do
    {ns, acc} = forward_reduce(ns, acc, fun)
    n = %Expr{n | nodes: ns}
    fun.(n.forward.(n), acc)
  end

  def forward_reduce([n | ns], acc, fun) do
    {ns, acc} = forward_reduce(ns, acc, fun)
    {n, acc} = forward_reduce(n, acc, fun)
    {[n | ns], acc}
  end

  def forward_reduce([], acc, _fun), do: {[], acc}

  @spec breadth_first_reduce(t, any, (t, any -> {t, any})) :: {t, any}
  def breadth_first_reduce(%Expr{nodes: ns} = n, acc, fun) do
    {n, acc} = fun.(n, acc)
    {[n], acc} = breadth_first_reduce([ns], acc, fun, [n])
    {n, acc}
  end

  def breadth_first_reduce([ns | rest_ns], acc, fun, [n | rest_n]) do
    {ns, acc} =
      Enum.reduce(ns, {[], acc}, fn n, {ns, acc} ->
        {n, acc} = fun.(n, acc)
        {[n | ns], acc}
      end)

    ns = Enum.reverse(ns)
    ns_ns = Enum.map(ns, & &1.nodes)

    {ns, acc} = breadth_first_reduce(ns_ns, acc, fun, ns)
    {rest_n, acc} = breadth_first_reduce(rest_ns, acc, fun, rest_n)

    {[%Expr{n | nodes: ns} | rest_n], acc}
  end

  def breadth_first_reduce([], acc, _fun, []) do
    {[], acc}
  end

  @spec backward(vector) :: vector
  def backward(n, grad \\ 1)
  def backward([n | ns], grad), do: [backward(n, grad) | backward(ns, grad)]
  def backward([], _grad), do: []
  def backward(n, grad), do: %Expr{n | grad: grad} |> backward_apply()

  defp backward_apply(%Expr{} = n) do
    n = n.backward.(n)
    %Expr{n | nodes: backward_apply(n.nodes)}
  end

  defp backward_apply([n | ns]), do: [backward_apply(n) | backward_apply(ns)]

  defp backward_apply([]), do: []

  @spec backward_reduce(vector, any, (t, any -> any)) :: any
  def backward_reduce(n, grad \\ 1, acc, fun),
    do: %Expr{n | grad: grad} |> backward_apply(acc, fun)

  defp backward_apply(%Expr{} = n, acc, fun) do
    n = n.backward.(n)
    acc = fun.(n, acc)
    backward_apply(n.nodes, acc, fun)
  end

  defp backward_apply([n | ns], acc, fun) do
    acc = backward_apply(n, acc, fun)
    backward_apply(ns, acc, fun)
  end

  defp backward_apply([], acc, _fun), do: acc

  @spec run(vector) :: vector
  def run(n, fun \\ & &1), do: n |> forward(fun) |> backward()

  @spec update(vector, label, (number -> number)) :: vector
  def update(n, label, fun) do
    transform_fun = fn
      %Expr{label: ^label} = n ->
        %Expr{n | value: map(n.value, &fun.(&1))}

      n ->
        n
    end

    forward(n, transform_fun) |> backward()
  end

  def map([x | xs], fun), do: [fun.(x) | map(xs, fun)]
  def map([], _fun), do: []
  def map(x, fun), do: fun.(x)

  def map([x | xs], [y | ys], fun), do: [fun.(x, y) | map(xs, ys, fun)]
  def map([], _y, _fun), do: []
  def map(_x, [], _fun), do: []
  def map([x | xs], y, fun), do: [fun.(x, y) | map(xs, y, fun)]
  def map(x, [y | ys], fun), do: [fun.(x, y) | map(x, ys, fun)]
  def map(x, y, fun), do: fun.(x, y)

  def sum(%Expr{} = n), do: n
  def sum(v), do: Enum.reduce(v, &add/2)

  @spec reduce(vector, any, (t, any -> any)) :: any
  def reduce(%Expr{nodes: ns} = n, acc, fun), do: reduce(ns, fun.(n, acc), fun)
  def reduce([n | ns], acc, fun), do: reduce(ns, reduce(n, acc, fun), fun)
  def reduce([], acc, _fun), do: acc

  def dot(x, y) do
    map(x, y, fn x, y -> map(x, y, &mul(&1, &2)) |> reduce(0, &(&2 + &1)) end)
  end

  @spec find_all(vector, label) :: [t]
  def find_all(n, label) do
    reduce(n, [], fn n, acc ->
      if n.label == label, do: [n | acc], else: acc
    end)
  end

  @spec grad(t | [t], label) :: [t]
  def grad(n, label) do
    reduce(n, 0, fn n, acc ->
      if n.label == label, do: acc + n.grad, else: acc
    end)
  end

  @spec size(t | [t]) :: [t]
  def size(n), do: reduce(n, 0, fn _, acc -> acc + 1 end)

  @spec add(value, value) :: t
  def add(n1, n2), do: new(:+, [n1, n2], &add_value/1, &add_grad/1)

  defp add_value(%Expr{nodes: [n1, n2]} = n) do
    %Expr{n | value: n1.value + n2.value}
  end

  defp add_grad(%Expr{nodes: [n1, n2]} = n) do
    n1 = %Expr{n1 | grad: n.grad}
    n2 = %Expr{n2 | grad: n.grad}

    %Expr{n | nodes: [n1, n2]}
  end

  @spec sub(value, value) :: t
  def sub(n1, n2), do: new(:-, [n1, n2], &sub_value/1, &sub_grad/1)

  defp sub_value(%Expr{nodes: [n1, n2]} = n) do
    %Expr{n | value: n1.value - n2.value}
  end

  defp sub_grad(%Expr{nodes: [n1, n2]} = n) do
    n1 = %Expr{n1 | grad: n.grad}
    n2 = %Expr{n2 | grad: -n.grad}

    %Expr{n | nodes: [n1, n2]}
  end

  @spec mul(value, value) :: t
  def mul(n1, n2), do: new(:*, [n1, n2], &mul_value/1, &mul_grad/1)

  defp mul_value(%Expr{nodes: [n1, n2]} = n) do
    %Expr{n | value: n1.value * n2.value}
  end

  defp mul_grad(%Expr{nodes: [n1, n2]} = n) do
    n1 = %Expr{n1 | grad: n2.value * n.grad}
    n2 = %Expr{n2 | grad: n1.value * n.grad}

    %Expr{n | nodes: [n1, n2]}
  end

  @spec div(value, value) :: t
  def div(n1, n2), do: new(:/, [n1, n2], &div_value/1, &div_grad/1)

  defp div_value(%Expr{nodes: [n1, n2]} = n) do
    %Expr{n | value: n1.value / n2.value}
  end

  defp div_grad(%Expr{nodes: [n1, n2]} = n) do
    n1 = %Expr{n1 | grad: n2.value * n.grad / n2.value ** 2}
    n2 = %Expr{n2 | grad: -n1.value * n.grad / n2.value ** 2}

    %Expr{n | nodes: [n1, n2]}
  end

  @spec pow(value, value) :: t
  def pow(n1, n2), do: new(:**, [n1, n2], &pow_value/1, &pow_grad/1)

  defp pow_value(%Expr{nodes: [n1, n2]} = n) do
    %Expr{n | value: n1.value ** round(n2.value)}
  end

  defp pow_grad(%Expr{nodes: [n1, n2]} = n) do
    n1 = %Expr{n1 | grad: n2.value * n1.value ** round(n2.value - 1) * n.grad}

    %Expr{n | nodes: [n1, n2]}
  end

  @spec neg(value) :: t
  def neg(n), do: new(:neg, [n], &neg_value/1, &neg_grad/1)

  defp neg_value(%Expr{nodes: [n1]} = n) do
    %Expr{n | value: -n1.value}
  end

  defp neg_grad(%Expr{nodes: [n1]} = n) do
    n1 = %Expr{n1 | grad: -n.grad}
    %Expr{n | nodes: [n1]}
  end

  @spec relu(value) :: t
  def relu(n), do: new(:relu, [n], &relu_value/1, &relu_grad/1)

  defp relu_value(%Expr{nodes: [n1]} = n) do
    value = if n1.value < 0, do: 0, else: n1.value
    %Expr{n | value: value}
  end

  defp relu_grad(%Expr{nodes: [n1]} = n) do
    grad = if n1.value > 0, do: n.grad, else: 0
    n1 = %Expr{n1 | grad: grad}

    %Expr{n | nodes: [n1]}
  end

  @spec absolute(value) :: t
  def absolute(n), do: new(:abs, [n], &abs_value/1, &abs_grad/1)

  defp abs_value(%Expr{nodes: [n1]} = n) do
    %Expr{n | value: abs(n1.value)}
  end

  defp abs_grad(%Expr{nodes: [n1]} = n) do
    grad = if n1.value >= 0, do: n.grad, else: -n.grad
    n1 = %Expr{n1 | grad: grad}
    %Expr{n | nodes: [n1]}
  end

  defp new(op, nodes, forward, backward) do
    %Expr{
      op: op,
      nodes: Enum.map(nodes, &normalize/1),
      forward: forward,
      backward: backward
    }
  end

  defp normalize(%Expr{} = n), do: n
  defp normalize(n) when is_number(n), do: value(n)
  defp normalize([n | ns]), do: [normalize(n) | normalize(ns)]
  defp normalize([]), do: []
end
