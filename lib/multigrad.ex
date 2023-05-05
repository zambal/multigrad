defmodule Multigrad do
  @moduledoc """
  Documentation for `Multigrad`.
  """

  alias Multigrad.{Activation, JobServer}

  @type model :: term
  @type loss :: number
  @type rate :: float
  @type predict_fun :: ([number] -> [number])
  @type opts :: Keyword.t()

  @type x :: [number]
  @type y :: [number]
  @type data :: Enumerable.t({x, y})
  @type batch_size :: pos_integer
  @type training_data :: Enumerable.t(term)

  @type num_of_in :: pos_integer
  @type num_of_out :: pos_integer
  @type activation :: :linear | :relu
  @type layer_def :: num_of_out | {num_of_out, activation}
  @type layer_defs :: [layer_def]
  @type definition :: [num_of_in | layer_defs]

  @type b :: [number]
  @type ws :: [[number]]
  @type parameters :: [{b, ws, activation}]

  @callback to_training_data(data, batch_size) :: training_data

  @callback from_config(definition, opts) :: model

  @callback from_parameters(parameters) :: model

  @callback to_parameters(model, opts) :: parameters

  @callback train({model, loss}, training_data, rate) :: {model, loss}

  def start_link(opts \\ []) do
    :gen_statem.start_link({:local, JobServer}, JobServer, opts, [])
  end

  def init, do: :gen_statem.call(JobServer, :init)

  def load(fun, opts \\ []) when is_function(fun, 1),
    do: :gen_statem.cast(JobServer, {:load, {fun, opts}})

  def build(definition, opts \\ []), do: :gen_statem.cast(JobServer, {:build, {definition, opts}})
  def train(opts \\ []), do: :gen_statem.call(JobServer, {:train, opts})
  def step, do: :gen_statem.call(JobServer, :step)
  def stop, do: :gen_statem.call(JobServer, :stop)
  def pause, do: :gen_statem.call(JobServer, :pause)
  def resume, do: :gen_statem.call(JobServer, :resume)
  def reset, do: :gen_statem.call(JobServer, :reset)
  def reload, do: :gen_statem.call(JobServer, :reload)
  def rebuild, do: :gen_statem.call(JobServer, :rebuild)
  def set_module(mod), do: :gen_statem.call(JobServer, {:set_module, mod})
  def get_model, do: :gen_statem.call(JobServer, :get_model)
  def put_model(model), do: :gen_statem.call(JobServer, {:put_model, model})
  def get_parameters, do: :gen_statem.call(JobServer, :get_parameters)
  def get_predict_fun, do: :gen_statem.call(JobServer, :get_predict_fun)
  def get_info, do: :gen_statem.call(JobServer, :get_info)
  def status, do: :gen_statem.call(JobServer, :status)
  def await_ready, do: :gen_statem.call(JobServer, :await_ready)
  def await_done, do: :gen_statem.call(JobServer, :await_done)

  def rand_number, do: :rand.uniform() * 2.0 - 1.0

  def to_predict_fun(params) do
    parameters =
      Enum.map(params, fn {b, ws, activation} ->
        Stream.zip(b, ws) |> Enum.map(fn {b, ws} -> {b, ws, Activation.afun(activation)} end)
      end)

    &predict_fun(&1, parameters)
  end

  defp predict_fun(xs, [params]) do
    layer_predict_fun(xs, params)
  end

  defp predict_fun(xs, [params | rest]) do
    xs
    |> layer_predict_fun(params)
    |> predict_fun(rest)
  end

  defp layer_predict_fun(xs, [{b, ws, afun} | ns]) do
    [afun.(b + map_sum(ws, xs, 0.0)) | layer_predict_fun(xs, ns)]
  end

  defp layer_predict_fun(_xs, []), do: []

  defp map_sum([w | ws], [x | xs], sum), do: map_sum(ws, xs, sum + w * x)
  defp map_sum([], _xs, sum), do: sum
  defp map_sum(_ws, [], sum), do: sum


  defp dot([w | ws], [x | xs], acc) do

  end
end
