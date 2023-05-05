defmodule Multigrad.JobServer do
  @behaviour :gen_statem

  require Logger

  alias __MODULE__

  defstruct model: nil,
            data: [],
            epoch: 0,
            rounds: 0,
            batch_size: 8,
            rate: 0.0,
            load: nil,
            build: nil,
            handler: nil,
            epochs_per_stats: 1,
            mod: Nxgrad

  @type t :: %JobServer{}

  @type state :: :initialized | :loaded | :build | :ready | :training | :paused | :done

  @type event ::
          :next_epoch
          | :init
          | {:load, {(Keyword.t() -> Multigrad.samples()), Keyword.t()}}
          | {:build, {Multigrad.definition(), Keyword.t()}}
          | {:train, Keyword.t()}
          | :step
          | :stop
          | :pause
          | :resume
          | :reset
          | :reload
          | :rebuild
          | {:set_module, module()}
          | {:set_handler, (t -> t)}
          | :get_model
          | {:put_model, Multigrad.model()}
          | :get_parameters
          | :get_predict_fun
          | :get_info
          | :status
          | :await_ready
          | :await_done

  def callback_mode, do: :state_functions

  def init(opts) do
    rounds = Keyword.get(opts, :rounds, 1)
    batch_size = Keyword.get(opts, :batch_size, 8)
    rate = Keyword.get(opts, :rate, -0.0001)
    handler = Keyword.get(opts, :handler)
    epochs_per_stats = Keyword.get(opts, :epochs_per_stats, 1)
    mod = Keyword.get(opts, :mod, Nxgrad)

    env = %JobServer{
      rounds: rounds,
      batch_size: batch_size,
      rate: rate,
      handler: handler,
      epochs_per_stats: epochs_per_stats,
      mod: mod
    }

    {:ok, :initialized, env}
  end

  def initialized({:call, from}, ev, env) when ev in [:step, :stop, :pause, :resume, :reload, :rebuild, :get_parameters, :get_predict_fun] do
    result(env, {:reply, from, {:error, {:state, :initialized}}})
  end

  def initialized({:call, from}, {:train, _}, env) do
    result(env, {:reply, from, {:error, {:state, :initialized}}})
  end

  def initialized(et, ev, env) do
    handle_event(et, ev, :initialized, env)
  end

  def loaded({:call, from}, ev, env) when ev in [:step, :stop, :pause, :resume, :set_module, :get_parameters, :get_predict_fun] do
    result(env, {:reply, from, {:error, {:state, :loaded}}})
  end

  def loaded({:call, from}, {:train, _}, env) do
    result(env, {:reply, from, {:error, {:state, :loaded}}})
  end

  def loaded(et, ev, env) do
    handle_event(et, ev, :loaded, env)
  end

  def build({:call, from}, ev, env) when ev in [:step, :stop, :pause, :resume, :set_module] do
    result(env, {:reply, from, {:error, {:state, :build}}})
  end

  def build({:call, from}, {:train, _}, env) do
    result(env, {:reply, from, {:error, {:state, :build}}})
  end

  def build(et, ev, env) do
    handle_event(et, ev, :build, env)
  end

  def ready(et, ev, env) do
    handle_event(et, ev, :ready, env)
  end

  def training({:call, from}, ev, env) when ev in [:reload, :rebuild, :set_module] do
    result(env, {:reply, from, {:error, {:load, :build, :state, :training}}})
  end

  def training({:call, from}, {ev, _}, env) when ev in [:train, :put_model] do
    result(env, {:reply, from, {:error, {:state, :training}}})
  end

  def training(_et, {ev, _}, env) when ev in [:load, :build] do
    result(env, [])
  end

  def training(et, ev, env) do
    handle_event(et, ev, :training, env)
  end

  def paused({:call, from}, ev, env) when ev in [:reload, :rebuild, :set_module] do
    result(env, {:reply, from, {:error, {:state, :paused}}})
  end

  def paused({:call, from}, {ev, _}, env) when ev in [:train, :put_model] do
    result(env, {:reply, from, {:error, {:state, :paused}}})
  end

  def paused(_et, {ev, _}, env) when ev in [:load, :build] do
    result(env, [])
  end

  def paused(et, ev, env) do
    handle_event(et, ev, :paused, env)
  end

  def done({:call, from}, ev, env) when ev in [:resume, :set_module] do
    result(env, {:reply, from, {:error, {:state, :done}}})
  end

  def done({:call, from}, {:train, _}, env) do
    result(env, {:reply, from, {:error, {:state, :done}}})
  end

  def done(et, ev, env) do
    handle_event(et, ev, :done, env)
  end

  @spec handle_event(:gen_statem.event_type(), event, state, t) :: :gen_statem.event_handler_result(state)

  # Training loop

  def handle_event(:info, :next_epoch, :training, %JobServer{model: model, epoch: n, rounds: n} = env) do
    {_, loss} = model
    Logger.info("job server: training finished, final loss is #{loss}")
    result(env, :done, [])
  end

  def handle_event(:info, :next_epoch, :training, env) do
    env = handle_epoch(env)
    result(env, :training, next: :next_epoch)
  end

  def handle_event(:info, :next_epoch, _state, env) do
    result(env, [])
  end

  # API

  def handle_event({:call, from}, :init, _state, env) do
    env = %JobServer{env | model: nil, data: nil, epoch: 0}
    result(env, :initialized, {:reply, from, :ok})
  end

  def handle_event(:cast, {:load, args}, state, env) do
    new_state = if state == :initialized, do: :loaded, else: :ready

    case handle_load(args, env) do
      {:ok, env} -> result(env, new_state, [])
      :error -> result(env, [])
    end
  end

  def handle_event(:cast, {:build, args}, state, env) do
    new_state = if state == :initialized, do: :build, else: :ready

    case handle_build(args, env) do
      {:ok, env} -> result(env, new_state, [])
      :error -> result(env, [])
    end
  end

  def handle_event({:call, from}, {:train, opts}, _state, env) do
    env = handle_training_opts(opts, env)
    result(env, :training, [{:reply, from, :ok}, next: :next_epoch])
  end

  def handle_event({:call, from}, :step, _state, env) do
    if env.epoch == env.rounds do
      result(env, :done, {:reply, from, :ok})
    else
      env = handle_epoch(env)
      result(env, :paused, {:reply, from, :ok})
    end
  end

  def handle_event({:call, from}, :stop, _state, env) do
    result(env, :done, {:reply, from, :ok})
  end

  def handle_event({:call, from}, :pause, _state, env) do
    result(env, :paused, {:reply, from, :ok})
  end

  def handle_event({:call, from}, :resume, _state, env) do
    result(env, :training, [{:reply, from, :ok}, next: :next_epoch])
  end

  def handle_event({:call, from}, :reset, _state, env) do
    result(env, :ready, {:reply, from, :ok})
  end

  def handle_event({:call, from}, :reload, state, env) do
    new_state = if state == :loaded, do: :loaded, else: :ready

    case handle_load(env.load, env) do
      {:ok, env} -> result(env, new_state, {:reply, from, :ok})
      :error -> result(env, {:reply, from, :error})
    end
  end

  def handle_event({:call, from}, :rebuild, state, env) do
    new_state = if state == :build, do: :build, else: :ready

    case handle_build(env.build, env) do
      {:ok, env} -> result(env, new_state, {:reply, from, :ok})
      :error -> result(env, {:reply, from, :error})
    end
  end

  def handle_event({:call, from}, {:set_module, mod}, _state, env) do
    env = %JobServer{env | mod: mod}
    result(env, {:reply, from, :ok})
  end

  def handle_event({:call, from}, {:set_handler, handler}, _state, env) do
    env = %JobServer{env | handler: handler}
    result(env, {:reply, from, :ok})
  end

  def handle_event({:call, from}, :get_model, _state, env) do
    result(env, {:reply, from, env.model})
  end

  def handle_event({:call, from}, {:put_model, model}, state, env) do
    new_state = if state == :initialized, do: :loaded, else: :ready
    env = %JobServer{env | model: {model, -1}}
    result(env, new_state, {:reply, from, :ok})
  end

  def handle_event({:call, from}, :get_parameters, _state, env) do
    {model, _loss} = env.model
    {definition, _opts} = env.build
    parameters = env.mod.to_parameters(model, definition: definition)
    result(env, {:reply, from, {:ok, parameters}})
  end

  def handle_event({:call, from}, :get_predict_fun, _state, env) do
    {model, _loss} = env.model
    {definition, _opts} = env.build
    parameters = env.mod.to_parameters(model, definition: definition)
    result(env, {:reply, from, Multigrad.to_predict_fun(parameters)})
  end

  def handle_event({:call, from}, :get_info, _state, env) do
    result(env, {:reply, from, Map.take(env, [:epoch, :rounds, :seed, :rate, :mod])})
  end

  def handle_event({:call, from}, :status, state, env) do
    result(env, {:reply, from, state})
  end

  def handle_event({:call, _from}, :await_ready, state, env) when state in [:initialized, :build, :loaded] do
    result(env, :postpone)
  end

  def handle_event({:call, from}, :await_ready, _state, env) do
    result(env, {:reply, from, true})
  end

  def handle_event({:call, from}, :await_done, :done, env) do
    result(env, {:reply, from, true})
  end

  def handle_event({:call, _from}, :await_done, _state, env) do
    result(env, :postpone)
  end

  # Private

  defp result(env, actions) do
    actions = handle_next(actions)
    {:keep_state, env, actions}
  end

  defp result(env, state, actions) do
    env =
      case state do
        :ready ->
          %JobServer{env | epoch: 0}

        _ ->
          env
      end

    actions = handle_next(actions)

    {:next_state, state, env, actions}
  end

  defp handle_next(actions) when is_list(actions) do
    case List.keytake(actions, :next, 0) do
      nil ->
        actions

      {{:next, op}, actions} ->
        next(op)
        actions
    end
  end

  defp handle_next(actions), do: actions

  defp next(op) do
    Process.send_after(self(), op, 0)
  end

  defp handle_epoch(env) do
    {model, loss} = env.mod.train(env.model, env.data, env.rate)
    env = %JobServer{env | model: {model, loss}, epoch: env.epoch + 1}
    env = if is_nil(env.handler), do: env, else: env.handler.(env)
    if rem(env.epoch, env.epochs_per_stats) == 0, do: print_stats(loss, env)
    env
  end

  defp handle_load({fun, opts}, env) do
    Logger.info("job server: start loading training data...")

    case fun.(opts) do
      {:ok, samples} ->
        count = samples |> List.flatten() |> length()
        data = env.mod.to_training_data(samples, env.batch_size)
        Logger.info("job server: training data loading finished with #{count} samples")

        {:ok, %JobServer{env | load: {fun, opts}, data: data}}

      {:error, e} ->
        Logger.error(
          "job server: stopping because load fun returned with error #{inspect(e)}"
        )

        :error
    end
  end

  defp handle_build({definition, opts}, env) do
    Logger.info("job server: start building model...")

    case env.mod.from_config(definition, opts) do
      {:ok, model} ->
        Logger.info("job server: building model finished")

        {:ok, %JobServer{env | build: {definition, opts}, model: {model, -1}}}

      {:error, e} ->
        Logger.error(
          "job server: stopping because build fun returned with error #{inspect(e)}"
        )

        :error
    end
  end

  defp handle_training_opts(opts, env) do
    opts = Keyword.take(opts, [:rounds, :rate])
    struct(env, opts)
  end

  defp print_stats(loss, env) do
    if is_atom(loss) do
      Logger.info("training: current loss #{loss}, epochs to go: #{env.rounds - env.epoch}")
    else
      loss = Nx.to_number(loss)

      Logger.info(
        "training: current loss #{:erlang.float_to_binary(loss, decimals: 8)}, epochs to go: #{env.rounds - env.epoch}"
      )
    end
  end
end
