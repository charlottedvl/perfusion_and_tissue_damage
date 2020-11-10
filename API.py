from eventmodule import eventhandler
import sys

if __name__ == "__main__":
    args = eventhandler.parse_args(sys.argv[1:])
    path = eventhandler.path_to_config(args['--patient'])
    model = eventhandler.read_model(path, args['--event'])

    k = next(iter(model))
    analysis = model[k]['type']

    if analysis == 'PERFUSION':
        from perfusion.API import API
        sys.exit(API(sys.argv[1:]).evaluate())

    if analysis == 'OXYGEN':
        from oxygen.API import API
        sys.exit(API(sys.argv[1:]).evaluate())

    sys.exit("No model was evaluated.")
