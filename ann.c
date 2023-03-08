#define LUA_LIB

#include <lua.h>
#include <lauxlib.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

static int
llayer_toarray(lua_State *L) {
	float * f = (float *)luaL_checkudata(L, 1, "ANN_LAYER");
	int n = lua_rawlen(L, 1) / sizeof(float);
	lua_createtable(L, n, 0);
	int i;
	for (i=0;i<n;i++) {
		lua_pushnumber(L, f[i]);
		lua_rawseti(L, -2, i+1);
	}
	return 1;
}

struct layer {
	int n;
	float *data;		// The input from the former layer
				// or the image itself (for the input layer)
};

static struct layer
checklayer(lua_State *L, int idx) {
	struct layer layer;
	layer.data = luaL_checkudata(L, idx, "ANN_LAYER");
	layer.n = lua_rawlen(L, idx) / sizeof(float);
	return layer;
}

// function init(self, image)
/*
 *	This function applies to the input layer only, to initialise its data
 *	with the pixels from the image.
 */
static int
llayer_init(lua_State *L) {
	struct layer layer = checklayer(L, 1);
	size_t sz = 0;
	const uint8_t * image = (const uint8_t *)luaL_checklstring(L, 2, &sz);
	if (sz != layer.n)	// Handles size mismatch
		return luaL_error(L, "Invalid image size %d != %d",
				  (int)sz, layer.n);
	int i;
	for (i=0;i<layer.n;i++) {
		layer.data[i] = image[i] / 255.0f;	// Convert range 0~255 
							// to 0~1
	}
	lua_settop(L, 1);
	return 1;
}

// function init_n(self, index)
/*
 *	This function will clean up all the data (set them to 0) the set
 *	layer.data[index] to 1.f
 */
static int
llayer_init_n(lua_State *L) {
	struct layer layer = checklayer(L, 1);
	int n = luaL_checkinteger(L, 2);
	if (n < 0 || n >= layer.n)
		return luaL_error(L, "Invalid n (%d)", n);
	memset(layer.data, 0, sizeof(float) * layer.n);	// Reset other slots
	layer.data[n] = 1.0f;
	lua_settop(L, 1);
	return 1;
}

// function max(self)
/*
 *	Find out the maximum value in a layer. Return its index and its
 *	proportion of all values.
 */
static int
llayer_max(lua_State *L) {
	struct layer layer = checklayer(L, 1);
	float m = layer.data[0];
	float s = m;	// The summary of all the values
	int idx = 0;
	int i;
	for (i=1;i<layer.n;i++) {
		if (layer.data[i] > m) {
			m = layer.data[i];
			idx = i;
		}
		s += layer.data[i];
	}
	lua_pushinteger(L, idx);
	lua_pushnumber(L, m / s);	// proportion
	return 2;
}

static int
llayer(lua_State *L) {
	int n = luaL_checkinteger(L, 1);
	float * f = (float *)lua_newuserdatauv(L, n * sizeof(*f), 0);
	memset(f, 0, sizeof(*f) * n);
	if (luaL_newmetatable(L, "ANN_LAYER")) {
		lua_pushvalue(L, -1);
		lua_setfield(L, -2, "__index");
		luaL_Reg l[] = {
			{ "toarray", llayer_toarray },
			{ "init", llayer_init },
			{ "init_n", llayer_init_n },
			{ "max", llayer_max },
			{ NULL, NULL },
		};
		luaL_setfuncs(L, l, 0);
	}
	lua_setmetatable(L, -2);
	return 1;
}

struct connection {
	int input_n;
	int output_n;
};

/*
 *	Convert an integer returned by rand() to float.
 *	The resulted float value is between 0 amd 1
 */
static inline float
randf() {
	float f = ((rand() & 0x7fff) + 1) / (float)0x8000;
	return f;
}

/*
 *	Normally-distributed
 */
static inline float
randnorm(float r1, float r2) {
	static const float PI = 3.1415927f;
	float x = sqrtf( -2.0 * logf ( r1 ) ) * cosf ( 2.0 * PI * r2 );
	return x;
}

// function randn(self)
/*
 * 	Initialise connections with random values.
 */
static int
lconnection_randn(lua_State *L) {
	float *f = (float *)luaL_checkudata(L, 1, "ANN_CONNECTION");
	int n = lua_rawlen(L, 1) / sizeof(float);
	float s = randf();
	int i;
	for (i=0;i<n;i++) {
		float r = randf();
		f[i] = randnorm(s, r);
		s = r;
	}
	return 0;
}

// function (self, delta)
/*
 *	Accumulate delta. Used for batched training.
 */
static int
lconnection_accumulate(lua_State *L) {
	float *base = (float *)luaL_checkudata(L, 1, "ANN_CONNECTION");
	int base_n = lua_rawlen(L, 1) / sizeof(float);
	float *delta = (float *)luaL_checkudata(L, 2, "ANN_CONNECTION");
	int delta_n = lua_rawlen(L, 2) / sizeof(float);
	if (base_n != delta_n)
		return luaL_error(L, "accumlate size mismatch");
	int i;
	for (i=0;i<base_n;i++) {
		base[i] += delta[i];
	}
	return 0;
}

// function (self,delta,eta)
/*
 *	Update connection strength
 */
static int
lconnection_update(lua_State *L) {
	float *base = (float *)luaL_checkudata(L, 1, "ANN_CONNECTION");
	int base_n = lua_rawlen(L, 1) / sizeof(float);
	float *delta = (float *)luaL_checkudata(L, 2, "ANN_CONNECTION");
	int delta_n = lua_rawlen(L, 2) / sizeof(float);
	if (base_n != delta_n)
		return luaL_error(L, "update size mismatch");
	float eta = luaL_checknumber(L, 3);
	int i;
	for (i=0;i<base_n;i++) {
		base[i] = base[i] - eta * delta[i];
	}
	return 0;
}

/*
 *	Get the offset of connections from output_idx
 *	Strength of connections from the same output_idx is stored in a row,
 *	The first output_n members are bias.
 */
static inline float *
weight(float *base, struct connection *c, int output_idx) {
	return base + c->input_n * output_idx + c->output_n;
}


static inline float *
bias(float *base, struct connection *c) {
	(void)c;
	return base;
}

static int
lconnection_dump(lua_State *L) {
	float *f = (float *)luaL_checkudata(L, 1, "ANN_CONNECTION");
	lua_getiuservalue(L, 1, 1);
	struct connection *c = (struct connection *)lua_touserdata(L, -1);
	float *b = bias(f, c);
	int i,j;
	for (i=0;i<c->output_n;i++) {
		float *w = weight(f, c, i);
		printf("[%d] BIAS %g ", i, b[i]);
		for (j=0;j<c->input_n;j++) {
			if (w[j] != 0)
				printf("%d:%g ", j, w[j]);
		}
		printf("\n");
	}
	return 0;
}

static int
lconnection_import(lua_State *L) {
	float *f = (float *)luaL_checkudata(L, 1, "ANN_CONNECTION");
	lua_getiuservalue(L, 1, 1);
	struct connection *c = (struct connection *)lua_touserdata(L, -1);
	luaL_checktype(L, 2, LUA_TTABLE);	// bias
	luaL_checktype(L, 3, LUA_TTABLE);	// weight
	int size_bias = lua_rawlen(L, 2);
	int size_weight = lua_rawlen(L, 3);
	if (size_bias != size_weight && size_bias != c->output_n)
		return luaL_error(L, "Invalid size");
	float *b = bias(f, c);
	int i,j;
	for (i=0;i<c->output_n;i++) {
		if (lua_rawgeti(L, 2, i+1) != LUA_TNUMBER)
			return luaL_error(L, "Invalid bias[%d]", i+1);
		b[i] = lua_tonumber(L, -1);
		lua_pop(L, 1);
	}
	for (i=0;i<c->output_n;i++) {
		if (lua_rawgeti(L, 3, i+1) != LUA_TTABLE)
			return luaL_error(L, "Invalid weight[%d]", i+1);
		int n = lua_rawlen(L, -1);
		if (n != c->input_n)
			return luaL_error(L, "Invalid weight_size[%d]", i+1);
		float *w = weight(f, c, i);
		for (j=0;j<n;j++) {
			if (lua_rawgeti(L, -1, j+1) != LUA_TNUMBER)
				return luaL_error(L, "Invalid weight[%d][%d]", i+1, j+1);
			w[j] = lua_tonumber(L, -1);
			lua_pop(L, 1);
		}
		lua_pop(L, 1);
	}
	return 0;
}


// function (x,y)
/*
 *	Create connections from a layer with x neurons to one with y neurons.
 */
static int
lconnection(lua_State *L) {
	struct connection * c = (struct connection *)lua_newuserdatauv(L, sizeof(*c), 0);
	c->input_n = luaL_checkinteger(L, 1);
	c->output_n = luaL_checkinteger(L, 2);
	size_t sz = (c->input_n * c->output_n + c->output_n) * sizeof(float);
		// x * y floats in total
	float * data = (float *)lua_newuserdatauv(L, sz, 1);
	lua_pushvalue(L, -2);
	lua_setiuservalue(L, -2, 1);
	memset(data, 0, sz);
	if (luaL_newmetatable(L, "ANN_CONNECTION")) {
		lua_pushvalue(L, -1);
		lua_setfield(L, -2, "__index");
		luaL_Reg l[] = {
			{ "randn", lconnection_randn },
			{ "accumulate", lconnection_accumulate },
			{ "update", lconnection_update },
			{ "dump", lconnection_dump },
			{ "import", lconnection_import },
			{ NULL, NULL },
		};
		luaL_setfuncs(L, l, 0);
	}
	lua_setmetatable(L, -2);
	return 1;
}

/*
 *	Sigmoid activation function
 *	sigmoid(x) = 1 / (1 + exp(-x))
 */
static inline float
sigmoid(float z) {
	return 1.0f / (1.0f + expf(-z));
}

/*
 *	Let phi(y) be the inverse function of y = sigmoid(x)
 *	y		= 1 / (1 + exp(-x))
 *	1 + exp(-x)	= 1 / y
 *	exp(-x)		= 1 / y - 1
 *	-x		= ln((1 - y)/ y)
 *	x		= ln(y / (y - 1))
 *	So phi(y) = ln(y / (y - 1))
 *	phi'(y) = y / (y - 1) * (y / (y - 1))'
 *	where	(y / (y - 1))' = - 1 / (y - 1)^2
 *	phi'(y)	= (y / (y - 1)) * (-1 / (y - 1)^2)
 *		= y * (1 - y)
 */
static inline float
sigmoid_prime(float s) {
	return s * (1-s);
}

// function (inputLayer, outputLayer, connections)
/*
 *	Calculate the inputs of outputLayer with data in inputLayer and
 *	connection strength in connections.
 */
static int
lfeedforward(lua_State *L) {
	struct layer input = checklayer(L, 1);
	struct layer output = checklayer(L, 2);
	float *f = (float *)luaL_checkudata(L, 3, "ANN_CONNECTION");
	lua_getiuservalue(L, 3, 1);
	struct connection *c = (struct connection *)lua_touserdata(L, -1);
	int i,j;

	/*
	 *	For each output (i)
	 */
	for (i=0;i<output.n;i++) {
		float s = 0;
		float *w = weight(f, c, i);	// Get the index of weight
		float *b = bias(f, c);		// index of bias
		for (j=0;j<input.n;j++) {
			s += input.data[j] * w[j];	// weighted sum
		}
		output.data[i] = sigmoid(s + b[i]);	// add sum to bias,
							// then apply
							// activation function
	}
	return 0;
}

/*
 *	The core part of back propagation
 */

// https://builtin.com/machine-learning/backpropagation-neural-network
static struct connection *
get_connection(lua_State *L, int idx) { lua_getiuservalue(L, idx, 1);
	struct connection *c = (struct connection *)lua_touserdata(L, -1);
	lua_pop(L, 1);
	return c;
}

// [Input]  --(nabla)-->  [result/Expect]
//
// delta := (result - expect) * sigmoid'(result)
// nabla_b := delta
// nabla_w := dot(delta, Input)

// function (input,result,expect,connction)
/*
 *	Initialise backpropagation for the output layer. Result is stored
 *	in delta.
 */
static int
lbackprop_last(lua_State *L) {
	struct layer input = checklayer(L, 1);
	struct layer result = checklayer(L, 2);
	struct layer expect = checklayer(L, 3);
	float *delta = (float *)luaL_checkudata(L, 4, "ANN_CONNECTION");
	struct connection *c = get_connection(L, 4);
	if (c->input_n != input.n || c->output_n != result.n) {
		return luaL_error(L, "Invalid output delta");
	}
	float *nabla_b = bias(delta, c);
	int i, j;
	/*
	 *	
	 */
	for (i=0;i<c->output_n;i++) {
		float *nabla_w = weight(delta, c, i);
		/*
		 *	result.data[i] - expect.data[i] is the error
		 */
		// cost derivative
		float d = (result.data[i] - expect.data[i]) *
			   sigmoid_prime(result.data[i]);
		nabla_b[i] = d;

		/*
		 *	Calculate delta by the contributions that each
		 *	connection committed to the error
		 */
		for (j=0;j<input.n;j++) {
			nabla_w[j] = d * input.data[j];
		}
	}
	return 0;
}

// [Input] --(nabla)--> [Z] --(delta_last/conn_output)-->
//
// delta := dot(conn_output_weight, delta_last) * sigmoid'(Z)
// nabla_b := delta
// nabla_w := dot(delta, Input)

// function (inputLayer, outputLayer, expect, delta)
/*
 *	Backpropagation for hidden layers. Result is stored in delta.
 */
static int
lbackprop(lua_State *L) {
	struct layer input = checklayer(L, 1); struct layer z = checklayer(L, 2);

	float *delta = (float *)luaL_checkudata(L, 3, "ANN_CONNECTION");
	struct connection *input_c = get_connection(L, 3);

	float *delta_last = (float *)luaL_checkudata(L, 4, "ANN_CONNECTION");
	struct connection *output_c = get_connection(L, 4);

	float *conn_output = (float *)luaL_checkudata(L, 5, "ANN_CONNECTION");

	if (input_c->output_n != output_c->input_n || input_c->output_n != z.n || input_c->input_n != input.n) {
		return luaL_error(L, "input/output mismatch");
	}
	float *nabla_b = bias(delta, input_c);
	int i, j;
	for (i=0;i<z.n;i++) {
		float *nabla_w = weight(delta, input_c, i);
		float d = 0;
		float *delta_last_b = bias(delta_last, output_c);
		for (j=0;j<output_c->output_n;j++) {
			d += delta_last_b[j] * weight(conn_output, output_c, j)[i];
		}
		d *= sigmoid_prime(z.data[i]);
		nabla_b[i] = d;
		for (j=0;j<input.n;j++) {
			nabla_w[j] = d * input.data[j];
		}
	}
	return 0;
}

LUAMOD_API int
luaopen_ann(lua_State *L) {
	luaL_checkversion(L);
	luaL_Reg l[] = {
		{ "layer" , llayer },
		{ "connection", lconnection },
		{ "feedforward", lfeedforward },
		{ "backprop", lbackprop },
		{ "backprop_last", lbackprop_last },
		{ NULL, NULL },
	};
	luaL_newlib(L, l);
	return 1;
}
