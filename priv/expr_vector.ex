defmodule Exgrad.Expr do
  alias Exgrad.Expr

  defstruct [:value, :grad, :forward, :backward, :nodes, :label, :op]

  @type t :: %Expr{
          value: nil | number | [number],
          grad: nil | number | [number],
          forward: expr_fun,
          backward: expr_fun,
          nodes: [t] | [[t]],
          label: nil | label,
          op: op
        }

  @type label :: String.Chars.t()
  @type expr_fun :: (t -> t)
  @type value :: number | t | [number | t]
  @type op :: :value | :+ | :- | :* | :/ | :** | :neg | :relu | :abs | :sum

  @type expr :: t | [t]

  defmacro __using__(_opts) do
    quote do
      alias Exgrad.Expr
      import Exgrad.Expr.Algebra, only: [expr: 1, run: 1, defn: 2]
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

  @spec forward(expr, expr_fun) :: expr
  def forward(%Expr{nodes: ns} = n, fun) do
    n = %Expr{n | nodes: forward(ns, fun)}
    fun.(n.forward.(n))
  end

  def forward([n | ns], fun) do
    [forward(n, fun) | forward(ns, fun)]
  end

  def forward([], _fun), do: []

  @spec forward_reduce(expr, any, (t, any -> {t, any})) :: {expr, any}
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

  @spec backward(expr) :: expr
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

  @spec backward_reduce(expr, any, (t, any -> any)) :: any
  def backward_reduce(n, grad \\ 1, acc, fun), do: %Expr{n | grad: grad} |> backward_apply(acc, fun)

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

  @spec run(expr) :: expr
  def run(n, fun \\ & &1), do: n |> forward(fun) |> backward()

  @spec update(expr, label, (number -> number)) :: expr
  def update(n, label, fun) do
    transform_fun = fn
      %Expr{label: ^label} = n ->
        %Expr{n | value: map(n.value, &fun.(&1))}

      n ->
        n
    end

    forward(n, transform_fun) |> backward()
  end

  @spec reduce(expr, any, (t, any -> any)) :: any
  def reduce(%Expr{nodes: ns} = n, acc, fun), do: reduce(ns, fun.(n, acc), fun)
  def reduce([n | ns], acc, fun), do: reduce(ns, reduce(n, acc, fun), fun)
  def reduce([], acc, _fun), do: acc

  @spec find_all(expr, label) :: [t]
  def find_all(n, label) do
    reduce(n, [], fn n, acc ->
      if n.label == label, do: [n | acc], else: acc
    end)
  end

  @spec grad(expr, label) :: [t]
  def grad(n, label) do
    reduce(n, nil, fn
      %Expr{label: ^label} = n, nil -> n.grad
      %Expr{label: ^label} = n, acc -> map(acc, n.grad, &(&1 + &2))
      _, acc -> acc
    end)
  end

  @spec size(expr) :: [t]
  def size(n), do: reduce(n, 0, fn _, acc -> acc + 1 end)

  @spec add(value, value) :: t
  def add(n1, n2), do: new(:+, [n1, n2], &add_value/1, &add_grad/1)

  defp add_value(%Expr{nodes: [n1, n2]} = n) do
    %Expr{n | value: map(n1, n2, fn x, y -> map(x.value, y.value, &(&1 + &2)) end)}
  end

  defp add_grad(%Expr{nodes: [n1, n2]} = n) do
    n1 = map(n1, n.grad, &%Expr{&1 | grad: &2})
    n2 = map(n2, n.grad, &%Expr{&1 | grad: &2})

    %Expr{n | nodes: [n1, n2]}
  end

  @spec sub(value, value) :: t
  def sub(n1, n2), do: new(:-, [n1, n2], &sub_value/1, &sub_grad/1)

  defp sub_value(%Expr{nodes: [n1, n2]} = n) do
    %Expr{n | value: map(n1, n2, fn x, y -> map(x.value, y.value, &(&1 - &2)) end)}
  end

  defp sub_grad(%Expr{nodes: [n1, n2]} = n) do
    n1 = map(n1, n.grad, &%Expr{&1 | grad: &2})
    n2 = map(n2, n.grad, &%Expr{&1 | grad: -&2})

    %Expr{n | nodes: [n1, n2]}
  end

  @spec mul(value, value) :: t
  def mul(n1, n2), do: new(:*, [n1, n2], &mul_value/1, &mul_grad/1)

  defp mul_value(%Expr{nodes: [n1, n2]} = n) do
    %Expr{n | value: map(n1, n2, fn x, y -> map(x.value, y.value, &(&1 * &2)) end)}
  end

  defp mul_grad(%Expr{nodes: [n1, n2]} = n) do
    n1 = map(n1, n2, n.grad, fn x, y, grad -> %Expr{x | grad: map(y.value, &(&1 * grad))} end)
    n2 = map(n1, n2, n.grad, fn x, y, grad -> %Expr{y | grad: map(x.value, &(&1 * grad))} end)

    %Expr{n | nodes: [n1, n2]}
  end

  @spec div(value, value) :: t
  def div(n1, n2), do: new(:/, [n1, n2], &div_value/1, &div_grad/1)

  defp div_value(%Expr{nodes: [n1, n2]} = n) do
    %Expr{n | value: map(n1, n2, fn x, y -> map(x.value, y.value, &(&1 / &2)) end)}
  end

  defp div_grad(%Expr{nodes: [n1, n2]} = n) do
    n1 = map(n1, n2, n.grad, fn x, y, grad -> %Expr{x | grad: map(y.value, &(&1 * grad / &1 ** 2))} end)
    n2 = map(n1, n2, n.grad, fn x, y, grad -> %Expr{y | grad: map(x.value, y.value, &(-&1 * grad / &2 ** 2))} end)

    %Expr{n | nodes: [n1, n2]}
  end

  @spec pow(value, value) :: t
  def pow(n1, n2), do: new(:**, [n1, n2], &pow_value/1, &pow_grad/1)

  defp pow_value(%Expr{nodes: [n1, n2]} = n) do
    %Expr{n | value: map(n1, n2, fn x, y -> map(x.value, y.value, &(&1 ** round(&2))) end)}
  end

  defp pow_grad(%Expr{nodes: [n1, n2]} = n) do
    n1 =
      map(n1, n2, n.grad, fn x, y, grad ->
        %Expr{x | grad: map(x.value, y.value, &(&2 * &1 ** round(&2 - 1) * grad))}
      end)

    %Expr{n | nodes: [n1, n2]}
  end

  @spec neg(value) :: t
  def neg(n), do: new(:neg, [n], &neg_value/1, &neg_grad/1)

  defp neg_value(%Expr{nodes: [n1]} = n) do
    %Expr{n | value: map(n1, fn x -> map(x.value, &(-&1)) end)}
  end

  defp neg_grad(%Expr{nodes: [n1]} = n) do
    n1 = map(n1, n.grad, &%Expr{&1 | grad: -&2})
    %Expr{n | nodes: [n1]}
  end

  @spec sum(value) :: t
  def sum(n), do: new(:sum, [n], &sum_value/1, &sum_grad/1)

  defp sum_value(%Expr{nodes: [n1]} = n) do
    %Expr{n | value: nsum(n1)}
  end

  defp sum_grad(%Expr{nodes: [n1]} = n) do
    n1 = map(n1, n.grad, &%Expr{&1 | grad: &2})
    %Expr{n | nodes: [n1]}
  end

  @spec relu(value) :: t
  def relu(n), do: new(:relu, [n], &relu_value/1, &relu_grad/1)

  defp relu_value(%Expr{nodes: [n1]} = n) do
    value = map(n1, fn x -> map(x.value, &if(&1 < 0, do: 0, else: &1)) end)
    %Expr{n | value: value}
  end

  defp relu_grad(%Expr{nodes: [n1]} = n) do
    n1 =
      map(n1, fn x ->
        grad = map(x.value, n.grad, &if(&1 > 0, do: &2, else: 0))
        %Expr{x | grad: grad}
      end)

    %Expr{n | nodes: [n1]}
  end

  @spec absolute(value) :: t
  def absolute(n), do: new(:abs, [n], &abs_value/1, &abs_grad/1)

  defp abs_value(%Expr{nodes: [n1]} = n) do
    %Expr{n | value: map(n1, fn x -> map(x.value, &abs(&1)) end)}
  end

  defp abs_grad(%Expr{nodes: [n1]} = n) do
    n1 =
      map(n1, fn x ->
        grad = map(x.value, n.grad, &if(&1 >= 0, do: &2, else: -&2))
        %Expr{x | grad: grad}
      end)

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
  defp normalize(n) when is_list(n), do: Enum.map(n, &normalize/1)

  def map([_ | _] = x, fun), do: Enum.map(x, fun)
  def map(x, fun), do: fun.(x)

  def map([_ | _] = x, [_ | _] = y, fun), do: vmap(x, y, fun)
  def map([_ | _] = x, y, fun), do: Enum.map(x, &fun.(&1, y))
  def map(x, [_ | _] = y, fun), do: Enum.map(y, &fun.(x, &1))
  def map(x, y, fun), do: fun.(x, y)

  defp vmap([x | xs], [y | ys], fun), do: [fun.(x, y) | vmap(xs, ys, fun)]
  defp vmap([], _ys, _fun), do: []
  defp vmap(_xs, [], _fun), do: []

  def map([_ | _] = x, [_ | _] = y, z, fun), do: vmap(x, y, z, fun)
  def map([_ | _] = x, y, z, fun), do: vmap(x, y, z, fun)
  def map(x, [_ | _] = y, z, fun), do: vmap(x, y, z, fun)
  def map(x, y, z, fun), do: fun.(x, y, z)

  defp vmap([x | xs], [y | ys], [z | zs], fun), do: [fun.(x, y, z) | vmap(xs, ys, zs, fun)]
  defp vmap([], _ys, _zs, _fun), do: []
  defp vmap(_xs, [], _zs, _fun), do: []
  defp vmap(_xs, _ys, [], _fun), do: []
  defp vmap(x, [y | ys], [z | zs], fun), do: [fun.(x, y, z) | vmap(x, ys, zs, fun)]
  defp vmap([x | xs], y, [z | zs], fun), do: [fun.(x, y, z) | vmap(xs, y, zs, fun)]
  defp vmap([x | xs], [y | ys], z, fun), do: [fun.(x, y, z) | vmap(xs, ys, z, fun)]

  defp nsum([_ | _] = x), do: Enum.reduce(x, 0, &(&2 + vsum(&1.value)))
  defp nsum(x), do: vsum(x.value)

  defp vsum([_ | _] = x), do: Enum.sum(x)
  defp vsum(x), do: x
end
