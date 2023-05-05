alias Exgrad.Expr
alias Exgrad.NN.{Layer, MLP, Neuron}

defimpl Inspect, for: Neuron do
  import Inspect.Algebra

  def inspect(n, opts) do
    concat([
      "neuron(b: ",
      Inspect.inspect(n.b, opts),
      ", w: ",
      Inspect.List.inspect(n.w, opts),
      ", #{n.nonlin})"
    ])
    |> group()
  end
end

defimpl Inspect, for: Layer do
  import Inspect.Algebra

  def inspect(l, opts) do
    concat(["layer", Inspect.List.inspect(l.neurons, opts)]) |> group()
  end
end

defimpl Inspect, for: MLP do
  import Inspect.Algebra

  def inspect(mlp, opts) do
    concat(["mlp", Inspect.List.inspect(mlp.layers, opts)]) |> group()
  end
end

defimpl Inspect, for: Expr do
  import Inspect.Algebra

  def inspect(%Expr{op: :value, value: v, label: nil, grad: nil}, opts) do
    Inspect.inspect(v, opts)
  end

  def inspect(%Expr{op: :value, value: v, label: nil, grad: g}, opts) do
    group(concat(Inspect.inspect(v, opts), "<#{g}>"))
  end

  def inspect(%Expr{op: :value, value: v, label: label, grad: nil}, opts) do
    group(concat([Inspect.inspect(v, opts), "[", to_string(label), "]"]))
  end

  def inspect(%Expr{op: :value, value: v, label: label, grad: g}, opts) do
    group(concat([Inspect.inspect(v, opts), "[", to_string(label), "]<", to_string(g), ">"]))
  end

  def inspect(%Expr{op: :neg, nodes: [n1], label: nil, grad: nil}, opts) do
    Inspect.inspect(-n1.value, opts)
  end

  def inspect(%Expr{op: :neg, nodes: [n1], label: nil, grad: g}, opts) do
    group(concat(Inspect.inspect(-n1.value, opts), "<#{g}>"))
  end

  def inspect(%Expr{op: :neg, nodes: [n1], label: label, grad: nil}, opts) do
    group(concat([Inspect.inspect(-n1.value, opts), "[", to_string(label), "]"]))
  end

  def inspect(%Expr{op: :neg, nodes: [n1], label: label, grad: g}, opts) do
    group(
      concat([Inspect.inspect(-n1.value, opts), "[", to_string(label), "]<", to_string(g), ">"])
    )
  end

  def inspect(%Expr{op: :relu, nodes: [n1], label: nil, grad: nil}, opts) do
    group(concat(["relu(", Inspect.inspect(n1, opts), ")"]))
  end

  def inspect(%Expr{op: :relu, nodes: [n1], label: nil, grad: g}, opts) do
    group(concat(["relu(", Inspect.inspect(n1, opts), ")<#{g}>"]))
  end

  def inspect(%Expr{op: :relu, nodes: [n1], label: label, grad: nil}, opts) do
    group(concat(["relu(", Inspect.inspect(n1, opts), ")[", to_string(label), "]"]))
  end

  def inspect(%Expr{op: :relu, nodes: [n1], label: label, grad: g}, opts) do
    group(
      concat(["relu(", Inspect.inspect(n1, opts), ")[", to_string(label), "]<", to_string(g), ">"])
    )
  end

  def inspect(%Expr{op: :abs, nodes: [n1], label: nil, grad: nil}, opts) do
    Inspect.inspect(abs(n1.value), opts)
  end

  def inspect(%Expr{op: :abs, nodes: [n1], label: nil, grad: g}, opts) do
    group(concat(Inspect.inspect(abs(n1.value), opts), "<#{g}>"))
  end

  def inspect(%Expr{op: :abs, nodes: [n1], label: label, grad: nil}, opts) do
    group(concat([Inspect.inspect(abs(n1.value), opts), "[", to_string(label), "]"]))
  end

  def inspect(%Expr{op: :abs, nodes: [n1], label: label, grad: g}, opts) do
    group(
      concat([
        Inspect.inspect(abs(n1.value), opts),
        "[",
        to_string(label),
        "]<",
        to_string(g),
        ">"
      ])
    )
  end

  def inspect(%Expr{op: op, nodes: [n1, n2], label: nil, grad: nil}, opts) do
    group(
      concat([
        "(",
        glue(
          glue(Inspect.inspect(n1, opts), " ", Atom.to_string(op)),
          " ",
          Inspect.inspect(n2, opts)
        ),
        ")"
      ])
    )
  end

  def inspect(%Expr{op: op, nodes: [n1, n2], label: nil, grad: g}, opts) do
    group(
      concat([
        "(",
        glue(
          glue(Inspect.inspect(n1, opts), " ", Atom.to_string(op)),
          " ",
          Inspect.inspect(n2, opts)
        ),
        ")<#{g}>"
      ])
    )
  end

  def inspect(%Expr{op: op, nodes: [n1, n2], label: label, grad: nil}, opts) do
    group(
      concat([
        "(",
        glue(
          glue(Inspect.inspect(n1, opts), " ", Atom.to_string(op)),
          " ",
          Inspect.inspect(n2, opts)
        ),
        ")[",
        to_string(label),
        "]"
      ])
    )
  end

  def inspect(%Expr{op: op, nodes: [n1, n2], label: label, grad: g}, opts) do
    group(
      concat([
        "(",
        glue(
          glue(Inspect.inspect(n1, opts), " ", Atom.to_string(op)),
          " ",
          Inspect.inspect(n2, opts)
        ),
        ")[",
        to_string(label),
        "]<",
        to_string(g),
        ">"
      ])
    )
  end
end
