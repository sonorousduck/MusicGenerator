import pickle
import numpy as np
from keras.utils import np_utils
from music21 import instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization



def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return network_input, normalized_input

def create_network(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(128, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True, recurrent_dropout=0.25))
    model.add(LSTM(256, return_sequences=True, recurrent_dropout=0.25))
    model.add(LSTM(512))
    model.add(Dropout(0.15))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.15))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(n_vocab, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.summary()

    model.load_weights('weights.hdf5')

    return model

def create_midi(prediction_output):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        # If the pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()  # Generates piano midi
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # If the pattern is a note instead
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano() # Generates piano music
            output_notes.append(new_note)

        offset += 0.5

        midi_stream = stream.Stream(output_notes)

        midi_stream.write('midi', fp="test_output.mid")


def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(network_input) - 1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output



def generate():
    """ Generate a midi file """

    with open("data/notes", "rb") as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # Get all pitch names
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)


if __name__ == "__main__":
    generate()

