#!/usr/bin/env python3
"""
Script to unwrap Bidirectional layers, prune and quantize a Keras model, producing a smaller "mini" variant and a TFLite-optimized "turbo" variant.

Usage:
  python optimize_model.py \
      --model_dir path/to/saved_model_or_model.keras \
      --output_dir optimized \
      --end_step 1000 \
      [--begin_step 0] [--final_sparsity 0.5] [--epochs 0]

This will:
  1) Load your model and replace each Bidirectional layer with separate forward/backward RNN layers
  2) Wrap only Conv2D and Dense layers for pruning
  3) (Optionally) fine-tune the pruned model
  4) Strip pruning wrappers and save a pruned Keras model
  5) Convert it to a quantized TFLite model
"""

import argparse
import os
import traceback
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.layers import Input, Concatenate, LSTM, Bidirectional
from tensorflow.keras.models import Model


def load_model(model_dir):
    """Load a Keras model from a .keras file or SavedModel directory (compile=False)."""
    return tf.keras.models.load_model(model_dir, compile=False)


def unwrap_bidirectional(model_to_clone):
    """
    Build a new model where each Bidirectional layer is replaced by
    two independent RNN layers (forward and backward) and their outputs concatenated.
    """
    if not isinstance(model_to_clone, Model):
        raise ValueError("Input must be a tf.keras.Model instance.")

    # Dictionary to map original layer outputs (tensor references) to new layer outputs (tensors)
    layer_output_map = {}

    # --- 1. Handle Model Inputs ---
    new_inputs = []
    input_layers = []  # Keep track of the new Input layers themselves

    if isinstance(model_to_clone.input, list):
        # Multiple inputs
        for i, inp in enumerate(model_to_clone.inputs):
            # Create new Input layer based on the spec of the original tensor
            new_input_layer = Input(
                type_spec=inp.type_spec, name=f"input_{i+1}_unwrapped"
            )
            new_inputs.append(new_input_layer)
            input_layers.append(new_input_layer)  # Store the layer
            # Map original input tensor's REFERENCE to new input tensor
            # <<< FIX: Use .ref() for the key >>>
            layer_output_map[inp.ref()] = new_input_layer
            print(
                f"Mapping input {i}: {inp} -> {new_input_layer} using ref {inp.ref()}"
            )
    else:
        # Single input
        new_input_layer = Input(
            type_spec=model_to_clone.input.type_spec,
            name=f"{model_to_clone.input.name}_unwrapped"
            if hasattr(model_to_clone.input, "name")
            else "input_1_unwrapped",
        )
        new_inputs.append(new_input_layer)
        input_layers.append(new_input_layer)  # Store the layer
        # <<< FIX: Use .ref() for the key >>>
        layer_output_map[model_to_clone.input.ref()] = new_input_layer
        print(
            f"Mapping input: {model_to_clone.input} -> {new_input_layer} using ref {model_to_clone.input.ref()}"
        )

    # --- 2. Iterate through Layers and Rebuild Graph ---
    new_layer_instances = {}  # Map original layer instance to new layer instance(s)

    # We need to iterate in topological order for the graph connections
    # model.layers gives a flat list, which might not be topological for complex graphs
    # A safer way for functional models is often to traverse node by node,
    # but iterating through layers usually works for sequential/simple functional.
    # Let's stick with model.layers for now, assuming a reasonable structure.

    processed_outputs = set(
        layer_output_map.keys()
    )  # Keep track of tensors already mapped

    for layer in model_to_clone.layers:
        # Skip input layers as they are handled by creating new Input() objects
        # Their output tensors are already in layer_output_map
        if isinstance(layer, tf.keras.layers.InputLayer):
            print(f"Skipping InputLayer: {layer.name}")
            continue

        print(f"\nProcessing layer: {layer.name} ({type(layer).__name__})")

        # Get the input tensor(s) for this layer from the map
        layer_inputs = []
        original_input_nodes = layer._inbound_nodes  # Get connections

        if not original_input_nodes:
            print(
                f"Warning: Layer {layer.name} has no inbound nodes in the standard list? Checking config or skipping..."
            )
            # Sometimes layers might be defined but not connected initially in complex setups
            # Or it could be the InputLayer which we should have skipped.
            # If this layer IS supposed to be connected, this indicates a potential issue.
            continue

        # Use the first inbound node, typical for non-shared layers in sequence
        # More complex models (shared layers) might need iterating over all nodes
        # or a more robust graph traversal method.
        node = original_input_nodes[0]

        # Get the original Keras Tensors that are inputs to this layer IN THIS NODE
        original_input_tensors = node.input_tensors

        # Find the corresponding NEW input tensors using the map
        input_tensor_list = (
            original_input_tensors
            if isinstance(original_input_tensors, (list, tuple))
            else [original_input_tensors]
        )

        for tensor in input_tensor_list:
            # <<< FIX: Use .ref() for lookup >>>
            tensor_ref = tensor.ref()
            if tensor_ref not in layer_output_map:
                # This error can happen if layer iteration order doesn't match graph topology
                # Or if the graph structure is complex (e.g., disconnected parts initially)
                raise ValueError(
                    f"Cannot find input tensor ref {tensor_ref} (for original tensor {tensor}) "
                    f"needed by layer {layer.name} in map. "
                    f"Keys available: {list(layer_output_map.keys())}. "
                    f"Graph structure might be too complex or layer iteration order issue."
                )
            layer_inputs.append(layer_output_map[tensor_ref])

        # If layer_inputs has only one element, pass it directly, else pass as list
        current_input = layer_inputs[0] if len(layer_inputs) == 1 else layer_inputs
        print(f"  Input tensor(s) mapped for {layer.name}: {current_input}")

        # --- The Core Logic: Handle Bidirectional LSTM ---
        if isinstance(layer, Bidirectional) and isinstance(layer.layer, LSTM):
            print(f"  Found Bidirectional LSTM: {layer.name}. Unwrapping...")
            bi_layer = layer
            lstm_layer = bi_layer.layer
            lstm_config = lstm_layer.get_config()
            bi_config = bi_layer.get_config()
            merge_mode = bi_config.get("merge_mode", "concat")

            if merge_mode != "concat":
                print(
                    f"Warning: Bidirectional layer {layer.name} has merge_mode='{merge_mode}'. "
                    f"Unwrapping currently only supports 'concat'. Proceeding with concat."
                )

            # Ensure return_sequences is consistent for both LSTMs
            return_sequences = lstm_config.get("return_sequences", False)
            # Copy config and update names/go_backwards
            forward_config = lstm_config.copy()
            forward_config["name"] = f"{lstm_layer.name}_forward"
            forward_config["go_backwards"] = False
            forward_config["return_sequences"] = return_sequences  # Ensure consistency

            backward_config = lstm_config.copy()
            backward_config["name"] = f"{lstm_layer.name}_backward"
            backward_config["go_backwards"] = True
            backward_config["return_sequences"] = return_sequences  # Ensure consistency

            # Create new layers
            forward_lstm = LSTM.from_config(forward_config)
            backward_lstm = LSTM.from_config(backward_config)
            concat_layer = Concatenate(axis=-1, name=f"{layer.name}_unwrapped_concat")

            # Connect the new layers
            forward_output = forward_lstm(current_input)
            backward_output = backward_lstm(current_input)
            new_output = concat_layer([forward_output, backward_output])

            print(f"    Created Forward LSTM: {forward_lstm.name}")
            print(f"    Created Backward LSTM: {backward_lstm.name}")
            print(f"    Created Concatenate: {concat_layer.name}")

            # Store the new layers for weight copying later
            new_layer_instances[layer] = {
                "forward": forward_lstm,
                "backward": backward_lstm,
                "concat": concat_layer,
            }

        # --- Handle Other Layers ---
        else:
            print(f"  Cloning layer: {layer.name}")
            # Get config and create a new layer instance
            config = layer.get_config()
            # Need to handle potential custom layer registration issues here if applicable
            try:
                new_layer = type(layer).from_config(config)
                # Set a unique name in case the original name conflicts in the new graph
                new_layer._name = f"{layer.name}_cloned"
            except Exception as e:
                print(f"Error creating layer {layer.name} from config: {e}")
                print(f"Config was: {config}")
                raise TypeError(
                    f"Failed to recreate layer {layer.name} ({type(layer).__name__}). "
                    f"Check if it's a custom layer requiring registration."
                ) from e

            # Connect the new layer
            new_output = new_layer(current_input)

            # Store the new layer instance
            new_layer_instances[layer] = new_layer

        # --- Update the output map ---
        # Map original output tensor REF(s) to new output tensor(s)
        original_output_tensors = node.output_tensors
        new_output_list = (
            new_output if isinstance(new_output, (list, tuple)) else [new_output]
        )
        original_output_list = (
            original_output_tensors
            if isinstance(original_output_tensors, (list, tuple))
            else [original_output_tensors]
        )

        if len(new_output_list) != len(original_output_list):
            raise TypeError(
                f"Output tensor count mismatch for layer {layer.name}. "
                f"Original: {len(original_output_list)}, New: {len(new_output_list)}"
            )

        for i, tensor in enumerate(original_output_list):
            # <<< FIX: Use .ref() for the key >>>
            tensor_ref = tensor.ref()
            layer_output_map[tensor_ref] = new_output_list[i]
            processed_outputs.add(
                tensor_ref
            )  # Mark this original tensor ref as processed
            print(
                f"  Mapping output {i}: {tensor} -> {new_output_list[i]} using ref {tensor_ref}"
            )

    # --- 3. Define Model Outputs ---
    new_outputs = []
    original_output_list_model = model_to_clone.outputs  # This is already a list
    for output_tensor in original_output_list_model:
        # <<< FIX: Use .ref() for lookup >>>
        output_ref = output_tensor.ref()
        if output_ref not in layer_output_map:
            raise ValueError(
                f"Cannot find original output tensor ref {output_ref} (for {output_tensor}) in map. "
                f"Available keys: {list(layer_output_map.keys())}"
            )
        new_outputs.append(layer_output_map[output_ref])

    # --- 4. Create the New Model ---
    # Use the new_inputs which are the actual Input Tensors created at the start
    new_model = Model(
        inputs=new_inputs, outputs=new_outputs, name=f"{model_to_clone.name}_unwrapped"
    )
    print(f"\nNew model created: {new_model.name}")

    # --- 5. Copy Weights ---
    print("\nCopying weights...")
    # It's generally safer to iterate through the *new* model's layers
    # and find the corresponding *original* layer to copy from, if possible.
    # However, the current structure maps original -> new, so we'll use that.

    # Create a reverse map for easier lookup if needed, or iterate originals
    # original_layer_map = {id(v): k for k, v in new_layer_instances.items() if not isinstance(v, dict)}
    # original_bi_layer_map = {id(v['forward']): k for k, v in new_layer_instances.items() if isinstance(v, dict)}
    # original_bi_layer_map.update({id(v['backward']): k for k, v in new_layer_instances.items() if isinstance(v, dict)})
    # (This reverse mapping might be complex if names aren't unique or layers are reused)

    # Let's stick to iterating original layers and finding their new counterparts
    for layer in model_to_clone.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if layer.get_weights() == []:  # Skip layers with no weights
            print(f"  Skipping {layer.name} (no weights)")
            continue

        if layer not in new_layer_instances:
            print(
                f"Warning: Original layer {layer.name} not found in new layer instances map. Skipping weight copy."
            )
            continue

        original_weights = layer.get_weights()
        target_layers_info = new_layer_instances[layer]

        if isinstance(target_layers_info, dict):  # It was a Bidirectional LSTM
            print(f"  Copying weights for unwrapped Bidirectional LSTM: {layer.name}")
            forward_lstm = target_layers_info["forward"]
            backward_lstm = target_layers_info["backward"]

            # Check if the original Bi-LSTM layer itself has weights (it should)
            if not original_weights:
                print(
                    f"    Warning: Original Bidirectional layer {layer.name} reported no weights. Cannot copy."
                )
                continue

            # Determine the split point for weights
            # Weights are [forward_kernel, forward_recurrent_kernel, forward_bias,
            #             backward_kernel, backward_recurrent_kernel, backward_bias]
            # Bias might be missing if use_bias=False
            expected_forward_weights_count = len(forward_lstm.get_weights())
            expected_backward_weights_count = len(backward_lstm.get_weights())

            if (
                len(original_weights)
                != expected_forward_weights_count + expected_backward_weights_count
            ):
                print(
                    f"    Warning: Weight count mismatch for Bidirectional {layer.name}. "
                    f"Original layer has {len(original_weights)} weights. "
                    f"Expected Forward LSTM ({forward_lstm.name}): {expected_forward_weights_count}, "
                    f"Expected Backward LSTM ({backward_lstm.name}): {expected_backward_weights_count}. "
                    "Weight copy might fail or be incorrect."
                )
                # Attempt standard split anyway if counts match, otherwise skip maybe?
                # Or check if bias is the difference: usually 3 weights per LSTM (kernel, recurrent, bias)
                split_point = expected_forward_weights_count  # Best guess
                if split_point >= len(original_weights):
                    print(
                        f"    Cannot determine split point. Skipping weights for {layer.name}"
                    )
                    continue  # Cannot proceed
            else:
                split_point = expected_forward_weights_count

            try:
                forward_weights = original_weights[:split_point]
                backward_weights = original_weights[split_point:]

                if len(forward_weights) == expected_forward_weights_count:
                    forward_lstm.set_weights(forward_weights)
                    print(f"    Weights set for {forward_lstm.name}")
                else:
                    print(
                        f"    Weight count mismatch after split for {forward_lstm.name}. Cannot set."
                    )

                if len(backward_weights) == expected_backward_weights_count:
                    backward_lstm.set_weights(backward_weights)
                    print(f"    Weights set for {backward_lstm.name}")
                else:
                    print(
                        f"    Weight count mismatch after split for {backward_lstm.name}. Cannot set."
                    )

            except Exception as e:
                print(f"    Error copying weights for unwrapped {layer.name}: {e}")
                traceback.print_exc()

        else:  # Standard layer clone
            new_layer = target_layers_info
            print(f"  Copying weights for layer: {layer.name} -> {new_layer.name}")
            try:
                if len(original_weights) == len(new_layer.get_weights()):
                    new_layer.set_weights(original_weights)
                else:
                    print(
                        f"    Warning: Weight count mismatch for {layer.name} -> {new_layer.name}. "
                        f"Original: {len(original_weights)}, New: {len(new_layer.get_weights())}. Cannot copy."
                    )
            except Exception as e:
                print(
                    f"    Error copying weights for {layer.name} -> {new_layer.name}: {e}"
                )

    print("\nCloning and unwrapping complete.")
    return new_model


def prune_model_by_layer(model, begin_step, end_step, final_sparsity):
    """
    Clone the model, applying pruning only to supported layers.
    """
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=final_sparsity,
            begin_step=begin_step,
            end_step=end_step,
        )
    }
    prune = tfmot.sparsity.keras.prune_low_magnitude

    def _apply_pruning(layer):
        if isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.Dense,
                tf.keras.layers.LSTM,
                tf.keras.layers.GRU,
            ),
        ):
            return prune(layer, **pruning_params)
        return layer

    return tf.keras.models.clone_model(model, clone_function=_apply_pruning)


def strip_pruning(pruned_model):
    """Strip pruning wrappers to obtain a final pruned model."""
    return tfmot.sparsity.keras.strip_pruning(pruned_model)


def convert_to_tflite(model, tflite_path):
    """Convert a Keras model to a quantized TFLite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)


def main():
    parser = argparse.ArgumentParser(
        description="Unwrap Bidirectional, prune and quantize a Keras model"
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to the saved Keras model directory or .keras file",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save optimized models"
    )
    parser.add_argument(
        "--begin_step",
        type=int,
        default=0,
        help="Global step at which to begin pruning",
    )
    parser.add_argument(
        "--end_step",
        type=int,
        required=True,
        help="Global step at which to end pruning",
    )
    parser.add_argument(
        "--final_sparsity",
        type=float,
        default=0.5,
        help="Final sparsity for pruning (0.0 - 1.0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Epochs to fine-tune during pruning (0 to skip)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for fine-tuning"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and unwrap the original model
    orig = load_model(args.model_dir)
    model = unwrap_bidirectional(orig)
    model.summary()
    model.save(
        "/home/tim/Documents/loghi-models/NA-HTR-CABR-M0003/unwrapped-model.keras"
    )

    # Wrap supported layers for pruning
    pruned = prune_model_by_layer(
        model,
        begin_step=args.begin_step,
        end_step=args.end_step,
        final_sparsity=args.final_sparsity,
    )

    pruned.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=model.loss,
        metrics=model.metrics,
    )

    if args.epochs > 0:
        print(f"Fine-tuning for {args.epochs} epochs...")
        # integrate your train_dataset here:
        # pruned.fit(train_dataset, epochs=args.epochs, batch_size=args.batch_size)

    final_model = strip_pruning(pruned)

    # Save pruned Keras
    pruned_path = os.path.join(args.output_dir, "model_pruned.keras")
    final_model.save(pruned_path)

    # Save TFLite
    tflite_path = os.path.join(args.output_dir, "model_optimized.tflite")
    convert_to_tflite(final_model, tflite_path)

    print(f"Pruned Keras model saved to: {pruned_path}")
    print(f"Optimized TFLite model saved to: {tflite_path}")


if __name__ == "__main__":
    main()
