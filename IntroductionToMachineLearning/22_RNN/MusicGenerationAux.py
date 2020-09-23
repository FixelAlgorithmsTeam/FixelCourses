import numpy             as np
import matplotlib.pyplot as plt
import torch.optim       as optim
import torch
import time

class Plot:
    def __init__(self, sTitle, sLabel, sXlabel, sColor, vData=[]):
        self.sTitle  = sTitle
        self.sLabel  = sLabel
        self.sXlabel = sXlabel
        self.sColor  = sColor
        self.vData   = vData

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class Recorder:
    def __init__(self, lPlots, figsize=(12,4)):
        self.lTitles = np.unique([oPlot.sTitle for oPlot in lPlots])
        self.N       = len(self.lTitles)
        self.fig, _  = plt.subplots(1, self.N, figsize=(12, 4))
        self.dAxes   = {}
        ii           = 0
        for oPlot in lPlots:
            ax = self.dAxes.get(oPlot.sTitle, None)
            if ax == None:
                ax                       = self.fig.axes[ii]
                ii                      += 1
                self.dAxes[oPlot.sTitle] = ax

            ax.set_title(oPlot.sTitle)
            ax.set_xlabel(oPlot.sXlabel)
            ax.plot(oPlot.vData, c=oPlot.sColor, label=oPlot.sLabel)
            ax.legend()
            ax.grid(True)

        plt.tight_layout()

    def Append(self, sTitle, sLabel, vData):
        ax = self.dAxes[sTitle]
        for oLine in ax.lines:
            if oLine.get_label() == sLabel:
                vYdata = np.append(oLine.get_ydata(), vData)
                N      = len(vYdata)
                oLine.set_data(list(range(N)), vYdata)
        lYlim = ax.axis()[2:4]
        if N > 1:
            ax.axis(xmin=0, xmax=N, ymin=np.minimum(np.min(vData), lYlim[0]), ymax=np.maximum(np.max(vData), lYlim[1]))
        else:
            ax.axis(xmin=0, xmax=N, ymin=np.min(vData), ymax=np.max(vData)+1e-10)

    def Get(self, sTitle, sLabel):
        ax = self.dAxes[sTitle]
        for oLine in ax.lines:
            if oLine.get_label() == sLabel:
                return oLine.get_ydata()

    def Draw(self):
        self.fig.canvas.draw()
        plt.pause(1e-10)


#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def Accuracy(mHatY, mY):
    mHatY = (mHatY > 0.5).float()
    return (mHatY == mY).float().mean().item()


from sklearn.metrics import r2_score

def R2Score(vHatY, vY):
    return r2_score(vY.view(-1).detach().cpu(), vHatY.view(-1).detach().cpu())

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
import time

def TrainLoop(oModel, oTrainDL, LossFunc, oOptim, oScheduler=None):

    epochLoss = 0
    epochAcc  = 0
    count     = 0                                #-- number of samples
    device    = next(oModel.parameters()).device #-- CPU\GPU
    oModel.train(True)

    #-- Iterate over the mini-batches:
    for ii, (mX, vY) in enumerate(oTrainDL):
        #-- Move to device (CPU\GPU):
        mX = mX.to(device)
        vY = vY.to(device)

        #-- Set gradients to zeros:
        oOptim.zero_grad()

        #-- Forward:
        mHatY = oModel(mX)
        loss  = LossFunc(mHatY, vY)

        #-- Backward:
        loss.backward()

        #-- Parameters update:
        oOptim.step()

        #-- Scheduler:
        stepFlag = False
        if isinstance(oScheduler, optim.lr_scheduler.CyclicLR):
            oScheduler.step()
            stepFlag = True

        if isinstance(oScheduler, OneCycleScheduler):
            oScheduler.step()
            stepFlag = True

        print(f'\rIteration: {ii:3d}: loss = {loss:.6f}', end='')
        #-- Accumulate loss:
        Nb         = mX[0].shape[0]
        epochLoss += Nb * loss.item()
        epochAcc  += Nb * Accuracy(mHatY, vY)
        count     += Nb

    print('', end='\r')
    epochLoss /= count
    epochAcc  /= count

    if oScheduler is not None and stepFlag == False:
        oScheduler.step()

    return epochLoss, epochAcc

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def ValidationLoop(oModel, oValDL, LossFunc):

    if oValDL is None:
        return 0, 0

    epochLoss = 0
    epochAcc  = 0
    count     = 0                                #-- number of samples
    device    = next(oModel.parameters()).device #-- CPU\GPU

    #-- Iterate over the mini-batches:
    oModel.train(False)
    with torch.no_grad():
        for ii, (mX, vY) in enumerate(oValDL):
            #-- Move to device (CPU\GPU):
            mX = mX.to(device)
            vY = vY.to(device)

            #-- Forward:
            mHatY = oModel(mX)
            loss  = LossFunc(mHatY, vY)

            Nb         = mX[0].shape[0]
            epochLoss += Nb * loss.item()
            epochAcc  += Nb * Accuracy(mHatY, vY)
            count     += Nb

    epochLoss /= count
    epochAcc  /= count

    return epochLoss, epochAcc

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def TrainModel(oModel, oTrainDL, oValDL, LossFunc, numEpochs, oOptim, oScheduler=None):

    oRecorder = Recorder([
        Plot('Loss',          'train', 'epoch', 'b'),
        Plot('Loss',          'val',   'epoch', 'r'),
        Plot('Accuracy',      'train', 'epoch', 'b'),
        Plot('Accuracy',      'val',   'epoch', 'r'),
        Plot('Learning rate', 'lr',    'epoch', 'b'),
    ])

    oRecorder.Append('Learning rate', 'lr', oOptim.param_groups[0]['lr']),

    bestAcc  = 0
    bestLoss = 1e20
    for epoch in range(numEpochs):

        startTime           = time.time()
        trainLoss, trainAcc = TrainLoop(oModel, oTrainDL, LossFunc, oOptim, oScheduler) #-- train
        valLoss,   valAcc   = ValidationLoop(oModel, oValDL, LossFunc)                  #-- validation

        #-- Display:
        oRecorder.Append('Loss',          'train', trainLoss),
        oRecorder.Append('Loss',          'val',   valLoss),
        oRecorder.Append('Accuracy',      'train', trainAcc),
        oRecorder.Append('Accuracy',      'val',   valAcc),
        oRecorder.Append('Learning rate', 'lr',    oOptim.param_groups[0]['lr']),
        oRecorder.Draw()

        endTime = time.time()
        print('Epoch '              f'{epoch:3d}:',                  end='')
        print(' | Train loss: '     f'{trainLoss:.5f}',              end='')
        print(' | Val loss: '       f'{valLoss:.5f}',                end='')
        print(' | Train Accuracy: ' f'{trainAcc:2.4f}',              end='')
        print(' | Val Accuracy: '   f'{valAcc:2.4f}',                end='')
        print(' | epoch time: '     f'{(endTime-startTime):3.3f} |', end='')

        #-- Save best model:
        if valLoss < bestLoss:
            bestLoss = valLoss
            try:
                torch.save(oModel.state_dict(), 'BestModelParameters.pt')
            except:
                pass
            print(' <-- Checkpoint!')
        else:
            print()

    oModel.load_state_dict(torch.load('BestModelParameters.pt'))

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def LearningRateSweep(oModel, oTrainDL, LossFunc, oOptim, vLearningRate):

    #-- Record mini-batches loss:
    oRecorder = Recorder([Plot('Batch loss', 'train', 'learning rate', 'b')])
    device    = next(oModel.parameters()).device #-- CPU\GPU

    numIter = len(vLearningRate)
    ii      = 0
    while ii < numIter:
        #-- Iterate over the mini-batches:
        for mX, vY in oTrainDL:
            if ii >= numIter:
                break

            #-- Move to device (CPU\GPU):
            mX = mX.to(device)
            vY = vY.to(device)

            #-- Set gradients to zeros:
            oOptim.zero_grad()

            #-- Forward:
            mHatY = oModel(mX)
#             print(mHatY.shape, vY.shape)
            loss  = LossFunc(mHatY, vY)

            #-- Backward:
            loss.backward()

            #-- Update parameters (with new learning rate)
            oOptim.param_groups[0]['lr'] = vLearningRate[ii]
            oOptim.step()

            oRecorder.Append('Batch loss', 'train', loss.item())
            oRecorder.Draw()

            ii += 1

    #-- Display:
    ax = oRecorder.dAxes['Batch loss']
    ax.lines[0].set_xdata(vLearningRate)
    ax.axis(xmin=vLearningRate[0], xmax=vLearningRate[-1])
    ax.set_xscale('log')
    oRecorder.Draw()

    return oRecorder

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class OneCycleScheduler:
    def __init__(self, oOptim, vLearningRate):
        self.oOptim = oOptim
        self.vLR    = vLearningRate
        self.tt     = 0

    def step(self):
        self.oOptim.param_groups[0]['lr'] = self.vLR[self.tt]
        self.tt += 1

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
#-- https://github.com/craffel/pretty-midi
import pretty_midi

#-- https://github.com/craffel/pretty-midi/blob/master/examples/reverse_pianoroll.py
def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def ReadMidi(fileName, Fs=10):
    oMidi = pretty_midi.PrettyMIDI(fileName, )
    mX    = (oMidi.get_piano_roll(fs=Fs)[20:108] > 1).astype(np.float)

    return mX

# def WriteMidi(fileName, mX, Fs=10):
#     L  = mX.shape[1]
#     mX = np.r_[np.zeros((20, L)), mX]

#     oPrettyMidi = piano_roll_to_pretty_midi(100 * (mX > 0), fs=Fs)
#     oPrettyMidi.write(fileName)


#-- https://www.pygame.org/news
#-- https://www.daniweb.com/programming/software-development/code/216976/play-a-midi-music-file-using-pygame
import pygame

def play_music(fileName):
    """
    stream music with mixer.music module in blocking manner
    this will stream the sound from disk while playing
    """
    clock = pygame.time.Clock()
    pygame.mixer.music.load(fileName)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        # check if playback has finished
        clock.tick(30)

def PlayMidi(mX, Fs=10):
    L  = mX.shape[1]
    mX = np.r_[np.zeros((20, L)), mX]

    fileName = '__TempMidi111__.mid'
    oPrettyMidi = piano_roll_to_pretty_midi(100 * (mX > 0), fs=Fs)
    oPrettyMidi.write(fileName)

    freq     = 44100  # audio CD quality
    bitsize  = -16    # unsigned 16 bit
    channels = 2      # 1 is mono, 2 is stereo
    buffer   = 1024   # number of samples
    pygame.mixer.init(freq, bitsize, channels, buffer)
    # optional volume 0 to 1.0
    pygame.mixer.music.set_volume(0.8)
    try:
        play_music(fileName)
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        # (works only in console mode)
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit
