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

import XCTest

@testable import TensorFlow

final class LayerTests: XCTestCase {
	func testBidirectionalGRU() {
		let inputSize = 2
		let gru = GRU<Float>(
      		GRUCell(
        		inputSize: inputSize,
        		hiddenSize: inputSize,
        		weightInitializer: glorotUniform(seed: (0xDeaf, 0xBeef)),
        		biasInitializer: zeros())
    	)
    	let bigru = BidirectionalGRU(inputSize: inputSize)

    	let x = Tensor<Float>(rangeFrom: 0.0, to: 0.2, stride: 0.1).rankLifted()
    	let inputs: [Tensor<Float>] = Array(repeating: x, count: 2) 

    	let (forwardOut, _) = valueWithPullback(at: gru, inputs) { gru, inputs in return gru(inputs) }
    	let (backwardOut, _) = valueWithPullback(at: gru, inputs) { gru, inputs in return gru(inputs.reversed()) }
    	let expected = zip(forwardOut, backwardOut).map { $0.hidden.concatenated(with: $1.hidden, alongAxis: 1) }

    	let output = bigru(inputs)

    	XCTAssertEqual(output, expected)
	}

	static var allTests = [
		("testBidirectionalGRU", testBidirectionalGRU)
	]
}
