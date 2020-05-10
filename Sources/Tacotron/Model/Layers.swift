// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow

/// A Bidirectional LSTM Layer with configurable merge function.
public struct Bidirectional: Layer {
  public var forward, backward: RecurrentLayer<LSTMCell<Float>>
  @noDerivative
  var mergeMode: String
 
  public init(_ cell: @autoclosure () -> LSTMCell<Float>, mergeMode: String = "cat"){
		precondition(["cat","sum","mul","ave"].contains(mergeMode), "Invalid mergeMode.")
		forward = RecurrentLayer(cell())
		backward = RecurrentLayer(cell())
		self.mergeMode = mergeMode
	}

	@differentiable
	public func merge(
		forward: Tensor<Float>, backward: Tensor<Float>
	) -> Tensor<Float> {
		switch self.mergeMode {
			case "sum":
				return forward + backward
			case "mul":
				return forward * backward
			case "ave":
				return (forward + backward) / 2
			default:
				return forward.concatenated(with: backward, alongAxis: 1)
		}
	}
	/// Parameters:
	///		- inputs: [Tensor<Float>], each of shape (batch, inputDims)
	///	Returns:
	///		- timeStepOutputs: [Tensor<Float>], each of shape (batch, inputDims)
	///		or (batch, 2 * inputDims) if mergeMode == "cat"
	@differentiable
	public func callAsFunction(_ inputs: [Tensor<Float>]) -> [Tensor<Float>] {
		precondition(!inputs.isEmpty, "'inputs' must be non-empty.")
		let lastIdx = withoutDerivative(at: inputs.count - 1)

		var forwardState = withoutDerivative(at:
		  forward.cell.zeroState(for: inputs[0]))

		var backwardState = withoutDerivative(at:
		  backward.cell.zeroState(for: inputs[lastIdx]))
		
		var timeStepOutputs: [Tensor<Float>] = []
	
		for i in 0 ... withoutDerivative(at: lastIdx) {
		  var forwardOutput = forward.cell(input: inputs[i],
			state: forwardState)
		  let backwardOutput = backward.cell(input: inputs[lastIdx - i],
			state: backwardState)
				forwardState = forwardOutput.state
		  backwardState = backwardOutput.state
		  forwardOutput.state.hidden = merge(
		  	forward: forwardOutput.state.hidden,
		  	backward: backwardOutput.state.hidden)
  		timeStepOutputs.append(forwardOutput.state.hidden)
		}	
		return timeStepOutputs
  }
}