local mnist = require "mnist"
local ann = require "ann"

local labels = mnist.labels "data/train-labels.idx1-ubyte"
local images = mnist.images "data/train-images.idx3-ubyte"

local network = {}	; network.__index = network

--[[
--	This function create a new network, with args.input inputs,
--		arg.output outputs and arg.hidden hidden neurons.
--	Connections are initialised randomly.
--]]
function network.new(args)
	--[[
	--	ih: input to hiden
	--	ho: hidden to output
	--	connection_*: the intension of connection
	--	*_layer: storing input of this layer
	--]]
	local n = {
		layer_n = { args.input, args.hidden, args.output },
		input_layer = ann.layer(args.input),
		connection_ih = ann.connection(args.input, args.hidden),
		hidden_layer = ann.layer(args.hidden),
		connection_ho = ann.connection(args.hidden, args.output),
		output_layer = ann.layer(args.output),
	}
	n.connection_ih:randn()
	n.connection_ho:randn()

	return setmetatable(n, network)
end

--[[
--	Do feedforwarding for all layers (input, hidden, output)
--]]
function network:feedforward(image)
	self.input_layer:init(image)
	ann.feedforward(self.input_layer, self.hidden_layer, self.connection_ih)
	ann.feedforward(self.hidden_layer, self.output_layer, self.connection_ho)

	return self.output_layer
end

--[[
--	Shffule the training data to avoid overstrabistics.
--	Swap pixels randomly.
--]]
local function shffule_training_data(t)
	local n = #t
	for i = 1, n - 1 do
		local r = math.random(i, n)
		t[i] , t[r] = t[r], t[i]
	end
end

--[[
--	The core part. Do pack propagation to train the network.
--]]
function network:train(training_data, batch_size, eta)
	shffule_training_data(training_data)

	local eta_ = eta / batch_size	-- factor influencing the speed of
					-- updating connection strength
	local delta_ih = ann.connection(self.layer_n[1], self.layer_n[2])
	local delta_ho = ann.connection(self.layer_n[2], self.layer_n[3])

	--[[
	--	Do backpropagation and save delta in delta_* (without _s suffix)
	--]]
	local function backprop(expect)
		ann.backprop_last(self.hidden_layer, self.output_layer,
				  expect, delta_ho)
		ann.backprop(self.input_layer, self.hidden_layer,
			     delta_ih, delta_ho, self.connection_ho)
	end

	--[[
	--	*_s stores the summary of delta.
	--]]
	local delta_ih_s = ann.connection(self.layer_n[1], self.layer_n[2])
	local delta_ho_s = ann.connection(self.layer_n[2], self.layer_n[3])

	for i = 1, #training_data, batch_size do
		-- We need to do feedforward first
		local r = self:feedforward(training_data[i].image)

		backprop(training_data[i].expect)

		delta_ih_s , delta_ih = delta_ih, delta_ih_s
		delta_ho_s , delta_ho = delta_ho, delta_ho_s

		--[[
		--	Accumulate delta, update after finishing all images
		--]]
		for j = 1, batch_size-1 do
			local image = training_data[i+j]
			if image then
				self:feedforward(image.image)
				backprop(training_data[i+j].expect)
				delta_ih_s:accumulate(delta_ih)
				delta_ho_s:accumulate(delta_ho)
			else
				eta_ = eta / j
				break
			end
		end

		--[[
		--	Update only once
		--]]
		self.connection_ih:update(delta_ih_s, eta_)
		self.connection_ho:update(delta_ho_s, eta_)
	end
end

--[[
--	Generate training data containing all images and their labels
--]]
local function gen_training_data()
	--[[
	--	result[i] is the expected output of an image labled number i
	--]]
	local result = {}
	for i = 0, 9 do
		result[i] = ann.layer(10)
		result[i]:init_n(i)
	end

	local training = {}	-- list of training data
	for i = 1, #images do
		training[i] = {
			image = images[i],
			expect = result[labels[i] ],
					-- now list result[] is used.
			value = labels[i],
		}
	end
	return training
end

local n = network.new {
	input = images.row * images.col,
	hidden = 30,
	output = 10,
}

local data = gen_training_data()

local labels = mnist.labels "data/t10k-labels.idx1-ubyte"
local images = mnist.images "data/t10k-images.idx3-ubyte"

local function test()
	local s = 0	-- Error counter
	for idx = 1, #labels do
		local r, p = n:feedforward(images[idx]):max()
		local label = labels[idx]
		if r~=label then
--			print(label, r, p)
			s = s + 1
		end
	end
	return s, #labels
end

for i = 1, 30 do
	n:train(data,10,3.0)
	print("Epoch", i, test())
end
