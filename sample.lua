--[[

This file samples characters from a trained model

Code is based on implementation in
https://github.com/oxford-cs-ml-2015/practical6

Changes (raymondhs):

Beam search are parallelized on GPU
UTF8 character handling

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

local stringx = require('pl.stringx')
utf8 = require 'lua-utf8'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',2000,'number of characters to sample')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:option('-beamsize',1,'defaults to 1')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

function splitstring(str, pattern)
  local t = {}
  for i in string.gfind(str, "([%z\1-\127\194-\244][\128-\191]*)") do
  --for i in str:gmatch(pattern) do
    t[#t + 1] = i
  end
  return t
end

-- gated print: simple utility function wrapping a print
function gprint(str)
  str = str:gsub('\n', '<n>')
  if opt.verbose == 1 then io.stderr:write(str .. '\n') end
  io:flush()
end

function clone_state(cc)
  new_state = {}
  if cc ~= nil then
    for L = 1,table.getn(cc) do
      -- c and h for all layers
      table.insert(new_state, cc[L]:clone())
    end
  else
    new_state = nil
  end
  return new_state
end

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 and opt.opencl == 0 then
  local ok, cunn = pcall(require, 'cunn')
  local ok2, cutorch = pcall(require, 'cutorch')
  if not ok then gprint('package cunn not found!') end
  if not ok2 then gprint('package cutorch not found!') end
  if ok and ok2 then
    gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
    gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    cutorch.manualSeed(opt.seed)
  else
    gprint('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
  end
end

-- check that clnn/cltorch are installed if user wants to use OpenCL
if opt.gpuid >= 0 and opt.opencl == 1 then
  local ok, cunn = pcall(require, 'clnn')
  local ok2, cutorch = pcall(require, 'cltorch')
  if not ok then print('package clnn not found!') end
  if not ok2 then print('package cltorch not found!') end
  if ok and ok2 then
    gprint('using OpenCL on GPU ' .. opt.gpuid .. '...')
    gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
    cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    torch.manualSeed(opt.seed)
  else
    gprint('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
  end
end

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
  gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

for c,i in pairs(vocab) do
  gprint(c .. ' ' .. i)
end

-- initialize the rnn state to all zeros
gprint('creating an ' .. checkpoint.opt.model .. '...')
local current_state
current_state = {}
for L = 1,checkpoint.opt.num_layers do
  -- c and h for all layers
  local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
  if opt.gpuid >= 0 and opt.opencl == 0 then h_init = h_init:cuda() end
  if opt.gpuid >= 0 and opt.opencl == 1 then h_init = h_init:cl() end
  table.insert(current_state, h_init:clone())
  if checkpoint.opt.model == 'lstm' then
    table.insert(current_state, h_init:clone())
  end
end

state_size = #current_state
default_state = clone_state(current_state)

-- Initialization
--[[
local seed_text = opt.primetext
gprint('missing seed text, using uniform probability over first character')
gprint('--------------------------')
prediction = torch.Tensor(1, #ivocab):fill(1)/(#ivocab)
if opt.gpuid >= 0 and opt.opencl == 0 then prediction = prediction:cuda() end
if opt.gpuid >= 0 and opt.opencl == 1 then prediction = prediction:cl() end
]]--

local seed_text = ''
local stexts = {}
while true do
  local line = io.read()
  if line == nil then break end
  seed_text = seed_text .. '\n' .. line
end
prediction = torch.Tensor(1, #ivocab):fill(1)/(#ivocab)
if opt.gpuid >= 0 and opt.opencl == 0 then prediction = prediction:cuda() end
if opt.gpuid >= 0 and opt.opencl == 1 then prediction = prediction:cl() end

-- start sampling/argmaxing
while (seed_text ~= '') do
  stext = seed_text
  seed_text = ''

  gprint('processing ' .. stext .. '...')
  chars = splitstring(stext, '.')

  beamsize = opt.beamsize
  beamState = {}
  beamScore = {}

  beamString = {} -- index to string
  beamLastChar = {}
  beamState[1] = clone_state(current_state)
  beamScore[1] = 0
  beamString[1] = ''

  ii = 1
  prev_char = nil
  while ii <= #chars do

    newBeamState = {}
    newBeamScore = {}
    newBeamString = {}
    newBeamLastChar = {}
    scores = {}
    beam_index = 1

    beamSize = 0
    for _, _ in pairs(beamState) do beamSize = beamSize+1 end
    prev_chars = torch.CudaTensor(beamSize)
    current_states = {}
    for i=1,state_size do
        table.insert(current_states, torch.CudaTensor(beamSize, checkpoint.opt.rnn_size))
    end

    cnt=1

    for cc, vv in pairs(beamState) do
      current_str = beamString[cc]
      current_state = vv
      current_score = beamScore[cc]

      strlen = utf8.len(current_str)
      if strlen > 0 then
        bb = beamLastChar[cc]
        if vocab[ bb ] == nil then
            prev_char = torch.Tensor{vocab["<unk>"]}
            prev_chars[cnt] = vocab["<unk>"]
        else
            prev_char = torch.Tensor{ vocab[ bb ] }
            prev_chars[cnt] = vocab[bb]
        end
      else
        prev_char = nil
      end
      for i=1,state_size do
        current_states[i][cnt]:copy(current_state[i])
      end

      cnt = cnt+1
    end

    cnt = 1

    new_states = {}
    predictions = nil
    if prev_char ~= nil then
      local lst = protos.rnn:forward{prev_chars, unpack(current_states)}
      for i=1,state_size do table.insert(new_states, lst[i]) end
      predictions = lst[#lst] -- last element holds the log probabilities
      predictions:div(opt.temperature) -- scale by temperature
    end

    for cc, vv in pairs(beamState) do
      current_str = beamString[cc]
      current_state = vv
      current_score = beamScore[cc]
      candidates = {chars[ii]}
      gprint('char(' .. chars[ii] .. ')')
      gprint('sco=' .. current_score .. ')')
      if utf8.upper(chars[ii]) ~= chars[ii] and vocab[chars[ii]] ~= nil then
        table.insert(candidates, utf8.upper(chars[ii]))
      end
      for jj = 1,#candidates do
        c = candidates[jj]
        this_char = torch.Tensor{vocab[c]}
        if vocab[c] == nil then this_char = torch.Tensor{vocab["<unk>"]} end
        
        if prev_char ~= nil then
          if this_char ~= nil then
            newstr = current_str .. c
            newsco = current_score + predictions[cnt][this_char[1]]
          end
          new_state = {}
          local state = torch.zeros(1, checkpoint.opt.rnn_size)
          for i=1,state_size do
            table.insert(new_state, state:clone():copy(new_states[i][cnt]))
          end

          gprint('\ttesting ' .. c .. ' score = ' .. predictions[cnt][this_char[1]] )
        else
          gprint('\ttesting ' .. c)
          new_state = default_state
          newstr = current_str .. c
          newsco = current_score
        end
        newBeamState[beam_index] = clone_state(new_state)
        newBeamScore[beam_index] = newsco
        newBeamString[beam_index] = newstr
        newBeamLastChar[beam_index] = c
        beam_index = beam_index + 1
        table.insert(scores, newsco)
      end
      cnt = cnt + 1
    end

    table.sort(scores)
    beamState = {}
    beamScore = {}
    beamString = {}
    beamLastChar = {}

    cnt = 0
    tid = #scores - beamsize + 1
    if tid<1 then tid = 1 end
    threshold = scores[tid]
    for cc,vv in pairs(newBeamScore) do
      gprint('\nBeam State:(' .. cc .. ',' .. vv .. ') threshold=' .. threshold .. ')')
      if vv >= threshold then
        beamState[cc] = newBeamState[cc]
        beamScore[cc] = newBeamScore[cc]
        beamString[cc] = newBeamString[cc]
        beamLastChar[cc] = newBeamLastChar[cc]
        cnt = cnt + 1
      end
    end
    ii = ii + 1
  end
  threshold = scores[#scores]
  beststr = nil
  for cc,vv in pairs(newBeamScore) do
    if vv == threshold then
      beststr = newBeamString[cc]
      current_state = newBeamState[cc]
    end
  end
  beststr = beststr:gsub("^%s+", ""):gsub("%s+$", "")
  io.write(beststr .. '\n') io.flush()
end

