defmodule Exgrad.Expr.Algebra do
  alias Exgrad.Expr

  defmacro __using__(_opts) do
    quote do
      import Kernel, except: [+: 2, -: 1, -: 2, *: 2, /: 2, **: 2, abs: 1]
      import unquote(__MODULE__), except: [epxr: 1, run: 1, defe: 2]
    end
  end

  defmacro expr(x) do
    quote do
      (fn ->
         use Exgrad.Expr.Algebra
         unquote(x)
       end).()
    end
  end

  defmacro run(x) do
    quote do
      unquote(x) |> Exgrad.Expr.Algebra.expr() |> Exgrad.Expr.run()
    end
  end

  defmacro defe(head, do: body) do
    quote do
      def unquote(head) do
        use Exgrad.Expr.Algebra
        unquote(body)
      end
    end
  end

  def v1 + v2 do
    Expr.add(v1, v2)
  end

  def -v1 do
    Expr.neg(v1)
  end

  def v1 - v2 do
    Expr.sub(v1, v2)
  end

  def v1 * v2 do
    Expr.mul(v1, v2)
  end

  def v1 / v2 do
    Expr.div(v1, v2)
  end

  def v1 ** v2 do
    Expr.pow(v1, v2)
  end

  defdelegate value(v1, label \\ nil), to: Expr
  defdelegate relu(v1), to: Expr
  defdelegate sum(v1), to: Expr
  defdelegate map(v1, fun), to: Expr
  defdelegate map(v1, v2, fun), to: Expr
  def abs(v1), do: Expr.absolute(v1)
end
