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

import Foundation
import TensorFlow

/// A highway net as defined in https://arxiv.org/abs/1505.00387
/// with configurable dense layers.
struct HighwayNet: Layer {
	var Hs: [Dense<Float>]
	var Ts: [Dense<Float>]
	/// Creates a highway network given layerSize and numLayers
	public init(layerSize: Int, numLayers: Int) {
		Hs = Array(repeating:
			Dense<Float>(inputSize: layerSize, outputSize: layerSize, activation: relu),
			count: numLayers)
		Ts = Array(repeating:
			Dense<Float>(inputSize: layerSize, outputSize: layerSize, activation: relu),
			count: numLayers)
	}

	@differentiable
	public func callAsFunction(_ inputs: Tensor<Float>) -> Tensor<Float> {
		var out = Hs[0](inputs) * Ts[0](inputs) + inputs * (1 - Ts[0](inputs))
		for i in 1 ..< withoutDerivative(at: Hs.count) {
			out = Hs[i](out) * Ts[i](out) + inputs * (1 - Ts[i](out))
		}
		return out 
	}
}

/// Preprocessing network
struct PreNet: Layer {
	var denses: [Dense<Float>] = []
	var dropOuts: [Dropout<Float>] = []
	/// Creates a preprocessing network given layerSizes and dropProbs
	public init(layerSizes: [(Int, Int)], dropProbs: [Double] = [0.5]) {
		precondition(dropProbs.count == layerSizes.count ||
			dropProbs.count == 1, "Invalid length of dropProbs")
		let drops = dropProbs.count == 1 ? 
			Array(repeating: dropProbs[0], count: layerSizes.count) :
			dropProbs
		for i in 0 ..< layerSizes.count {
			denses.append(Dense(inputSize: layerSizes[i].0,
				outputSize: layerSizes[i].1, activation: relu))
			dropOuts.append(Dropout(probability: drops[i]))
		}
	}

	@differentiable
	public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
		var out = dropOuts[0](denses[0](input))
		for i in 1 ..< withoutDerivative(at: denses.count){
			out = dropOuts[i](denses[i](out))
		}
		return out
	}
}

/// CBHG module as described in https://arxiv.org/pdf/1703.10135.pdf
struct CBHG: Layer {
	/// Number of sets of 1-D convolutional filters
	@noDerivative
	let K: Int
	var convLayers: [Conv1D<Float>]
	var maxPool: MaxPool1D<Float>
	var projLayers: [Conv1D<Float>]
	var highway: HighwayNet
	var bidirectional: Bidirectional
	/// Creates a CBHG module 
	public init () {
		
	}
	
}