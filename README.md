# Queen Feeder

A small pet project to try to implement a neural network that is capable of playing chess

### Idea

Initial approach was to simply take a board as input $X$, and try to predict label in range $[0;4096]$. Why this range? Because it encodes the information about `from_square` and `to_square` using the following formula:

$$
position_{from} * 64 + position_{to}
$$

Unfortunately, it didn't work well (especially since the output space is too big). That's when I found a [ConvChess](https://cs231n.stanford.edu/reports/2015/pdfs/ConvChess.pdf) paper from Stanford which describes a better architecture using convolutional layers & having different neural nets for each piece type (queen, pawn, bishop, etc)

At the moment of implementation, I am not really familiar with convolutional layers, so I left them for future iterations. However, I used the "different NNs per piece type" approach, which decreased the output space from 4096 to 64, and drastically increased model performance

### Project structure

Project consists of 2 main parts: 

- `train.ipynb` - used to preprocess the data & train the model
- `play.ipynb` - notebook that loads trained models, renders chess board in SVG & does the move based on model prediction

There are also some additional `.py` files that are common for both notebooks (e.g model definitions, constants, some preprocessing parts)

For the training I used the following dataset that has around 6.2 million games -> https://www.kaggle.com/datasets/arevel/chess-games

Since I am training only for white pieces, I filtered the dataset to only contain games where: white won & players ELO is > 1600

### Playing

When running the `play.ipynb` notebook you will play as black. In order to enter you move, you should first enter square from (e.g 51) and after you should enter square to (e.g 35). The following image should be helpful 

![square indexing](https://quantumai.google/cirq/experiments/unitary/quantum_chess/images/chess_board_indices.png)

### Testing

For testing purposes I created a [fresh account](https://www.chess.com/member/queen_feeder_1_0) and played 6 games

The model won 2 games, made 1 draw (I offered it to opponent because I had an advantage in terms of pieces, but had too little time), and lost 3 games. In one of the lost games I was 1 move away from a mate, but I ran out of time because I had to manually enter each move

> there are some other lost games on that account (with 0 moves), because I was assigned to play for black side, but at that particular moment I only trained the model based on white moves, so I had to resign

Generally it's pretty good at opennings, however the closer we get to endgame - the worse it becomes

### TODO

- Convolutional layers
- Find / generate a dataset which has more endgame positions (especially mates)
- Use "flipping" techinque, to train on both white & black moves (for the sake of simplicity now I only support NN playing as white)