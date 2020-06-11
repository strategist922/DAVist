import argparse
import os


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--option', type=str, default='train', help='train | test')
    parser.add_argument('--challenge', action='store_true')
    parser.add_argument('--no_tb', action='store_true')
    parser.add_argument('--log_metrics', action='store_false')
    parser.add_argument('--id', type=str, default='default', help='an id identifying this run/job')
    parser.add_argument('--vocab_threshold', type=int, default=3, help='vocab count')

    # Data input settings
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--sis_path', type=str,
                        default='data/sis/test.story-in-sequence.json')
    parser.add_argument('--reference_dir', type=str,
                        default='data/reference/')
    # parser.add_argument('--data_dir', type=str, default='../datasets/VIST')
    parser.add_argument('--desc_h5', type=str, default='VIST/description.h5')
    parser.add_argument('--features_h5', type=str,
                        default='data/frcnn_features/features.hdf5')
    parser.add_argument('--features_h5_index', type=str,
                        default='data/frcnn_features/imgid2idx.pkl')
    parser.add_argument('--story_h5', type=str, default='VIST/story.h5')
    parser.add_argument('--full_story_h5', type=str, default='VIST/full_story.h5')
    parser.add_argument('--story_line_json', type=str, default='VIST/story_line.json')
    parser.add_argument('--embedding_file', type=str, default='VIST/embedding.pt')
    parser.add_argument('--resume_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)""")
    parser.add_argument('--start_from_model', type=str, default=None)
    parser.add_argument('--test_iter', type=int, default=None)

    # Model settings
    parser.add_argument('--model', type=str, default="BaseModel")
    parser.add_argument('--task', type=str, default="story_telling", help='story_telling')
    parser.add_argument('--rnn_type', type=str, default='gru', help='gru, lstm, transformer')
    parser.add_argument('--visual_rnn_type', type=str, default='gru', help='gru, lstm, transformer')
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the RNN')
    parser.add_argument('--num_layers_decoder', type=int, default=1, help='number of layers in the decoder RNN')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='number of heads in the transformer')
    parser.add_argument('--word_embed_dim', type=int, default=300,
                        help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--feat_size', type=int, default=2048, help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--conv_feat_size', type=int, default=2048, help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--use_conv', action='store_true')
    parser.add_argument('--use_obj', action='store_false')
    parser.add_argument('--use_spatial', action='store_false')
    parser.add_argument('--use_classes', action='store_false')
    parser.add_argument('--use_attrs', action='store_false')
    parser.add_argument('--from_pretrained', action='store_false')
    parser.add_argument('--story_size', type=int, default=5,
                        help='number of images/sentences appearing in each story')
    parser.add_argument('--num_obj', type=int, default=36,
                        help='number of butd objects')
    parser.add_argument('--num_classes', type=int, default=1601,
                        help='number of butd objects')
    parser.add_argument('--num_attrs', type=int, default=400,
                        help='number of butd objects')
    parser.add_argument('--spatial_size', type=int, default=6,
                        help='number of butd objects spatial features')
    parser.add_argument('--bad_valid_threshold', type=int, default=5,
                        help='how many valid runs to wait before decaying the learning rate')
    parser.add_argument('--with_position', action='store_true',
                        help='whether to use position embedding for the image feature')
    parser.add_argument('--transform_q', action='store_true',
                        help='transform query in attention layer')
    parser.add_argument('--no_local_attention', action='store_true',
                        help='use local attention')
    parser.add_argument('--local_attention_type', type=str, default='dot')
    parser.add_argument('--visual_transformer', action='store_true')


    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='set to True to have the data reshuffled at every epoch during training ')
    parser.add_argument('--batch_size', type=int, default=64, help='minibatch size')
    parser.add_argument('--workers', type=int, default=8,
                        help='number of workers to load data')
    parser.add_argument('--grad_clip', type=float, default=10,
                        help='clip gradients at this value')
    parser.add_argument('--visual_dropout', type=float, default=0.2,
                        help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='strength of dropout in the Language Model RNN')
    parser.add_argument('--beam_size', type=int, default=1,
                        help='indicates number of beams in beam search. This is only used in the evaluation mode')

    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='Adam',
                        help='RMSprop|SGD|momSGD|Adam|Adadelta|YF')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1,
                        help='from which epoch to start decaying learning rate? (-1 = dont)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=5,
                        help='every how many epochs thereafter to drop LR')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8,
                        help='decay parameter for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum, only used in rmsprop & sgd')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    # Adam optimizer setting
    parser.add_argument('--optim_alpha', type=float, default=0.8, help='alpha for momentum')
    parser.add_argument('--optim_beta', type=float, default=0.999, help='beta for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                        help='epsilon that goes into denominator for smoothing')

    # Reinforcement learning
    parser.add_argument('--start_rl', type=int, default=-1,
                        help='at what epoch to start RL training, -1 means never')
    parser.add_argument('--reward_type', type=str, default='METEOR',
                        help="CIDEr | ROUGE_L | METEOR | Bleu_4 | Bleu_3")
    parser.add_argument('--rl_weight', type=float, default=0.5)
    parser.add_argument('--cached_tokens', type=str, default='VIST/VIST-train-words',
                        help='used to compute CIDEr reward')
    parser.add_argument('--use_feature_with_context', type=bool, default=False)
    parser.add_argument('--decoding_method_DISC', type=str, default='sample', help='greedy | sample')

    # Schedule sampling
    parser.add_argument('--scheduled_sampling_start', type=int, default=0,
                        help='at what epoch to start decay gt probability, -1 means never')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=4,
                        help='every how many epochs to increase scheduled sampling probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                        help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                        help='Maximum scheduled sampling prob.')

    # Evaluation/Checkpointing
    parser.add_argument('--metric', type=str, default='METEOR',
                        help="XE | CIDEr | ROUGE_L | METEOR | Bleu_4 | Bleu_3")
    parser.add_argument('--save_checkpoint_every', type=int, default=1000,
                        help='how often to save a model checkpoint (in iterations)')
    parser.add_argument('--checkpoint_path', type=str, default='data/save',
                        help='directory to store checkpointed models')
    parser.add_argument('--losses_log_every', type=int, default=10,
                        help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=bool, default=True,
                        help='whether to load previous best score when resuming training.')
    parser.add_argument('--log_step', type=int, default=20,
                        help='how often to log training loss')
    parser.add_argument('--prefix', type=str, default='fc/', help="feature folder")
    parser.add_argument('--penalty', type=float, default=4, help="sampling penalty")

    # misc
    parser.add_argument('--always', type=str, default=None, help='always train one model, no alternating training')
    parser.add_argument('--D_iter', type=int, default=100, help='Discriminator update iterations')
    parser.add_argument('--G_iter', type=int, default=100, help='Generator update iterations')
    parser.add_argument('--activation', type=str, default="sign",
                        help='the last activation function of the reward model: sign | tahn')
    parser.add_argument('--challenge_dir', type=str,
                        default='/home/ubuntu/VisualStorytelling/VIST-Challenge-NAACL-2018/runnable_jar/EvalVIST.jar')
    parser.add_argument('--save_code', type=bool, default=True,
                        help='whether to save a copy of the code to folder.')

    args = parser.parse_args()

    args.desc_h5 = args.desc_h5.replace('VIST', f'VIST/count{args.vocab_threshold}')
    args.story_h5 = args.story_h5.replace('VIST', f'VIST/count{args.vocab_threshold}')
    args.full_story_h5 = args.full_story_h5.replace('VIST', f'VIST/count{args.vocab_threshold}')
    args.story_line_json = args.story_line_json.replace('VIST', f'VIST/count{args.vocab_threshold}')
    args.embedding_file = args.embedding_file.replace('VIST', f'VIST/count{args.vocab_threshold}')

    if args.start_rl >= 0:
        args.metric = args.reward_type

    return args
