defmodule Exgrad.Expr.Cytoscape do
  use Exgrad.Expr

  def to_graph(n, path) do
    {n, _} =
      Expr.breadth_first_reduce(n, 0, fn n, id ->
        n = Expr.value(n, "#{id}||#{n.label}")
        {n, id + 1}
      end)

    {_, elements} =
      Expr.breadth_first_reduce(n, [], fn n, acc ->
        [id, label] = String.split(n.label, "||")

        value =
          if is_number(n.grad) do
            "#{inspect(n.value)}<#{inspect(n.grad)}>"
          else
            inspect(n.value)
          end

        if n.op == :value do
          {n, acc ++ [%{data: %{id: id, label: "#{value} #{label}"}}]}
        else
          edges =
            Enum.map(n.nodes, fn t ->
              [tid, _label] = String.split(t.label, "||")
              %{data: %{id: "#{id}:#{tid}", source: id, target: tid}}
            end)

          {n, acc ++ [%{data: %{id: id, label: "#{n.op} (#{value}) #{label}"}} | edges]}
        end
      end)

    IO.inspect elements

    json = Jason.encode!(elements)

    template_path = Application.app_dir(:multigrad, "priv/cytoscape_template.html.eex")

    {bin, _} =
      EEx.compile_file(template_path)
      |> Code.eval_quoted(elements: json)

    File.write!(path, bin)
  end
end
