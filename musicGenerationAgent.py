import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


def prepare_sequences(notes, n_vocab):

    sequence_length = 100

    pitchnames = sorted(set(item for item in notes))

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Makes the input into a form that LSTMs can understand

    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # Normalize
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output


def train_network():
    notes = get_notes()

    # This gives the amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def get_notes():
    notes = []

    for file in glob.glob("music/*.midi"):
        midi = converter.parse(file)


        try: # This is if the file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes


        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

        with open('data/notes', 'wb') as filepath:
            pickle.dump(notes, filepath)

        return notes



def create_network(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(LSTM(512))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(n_vocab, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.summary()
    return model

def train(model, network_input, network_output):
    filepath = "weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )

    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=2000, batch_size=128, callbacks=callbacks_list)
    model.save('weights.hdf5')

if __name__ == "__main__":
    # get_notes()
    train_network()
