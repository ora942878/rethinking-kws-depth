"""Run the fixed paper training, experiments and figure/table generation."""
import importlib,sys
COMMANDS={'train':'train.run','cache-kwt':'experiments.train.prepare_kwt_cache',
    'prepare':'experiments.export','evaluate':'experiments.evaluate','budget':'experiments.budget',
    'inference':'experiments.inference','paper':'paper.run'}
if len(sys.argv)<2 or sys.argv[1] in ('-h','--help'):
    print('Usage: python run.py <'+ '|'.join(COMMANDS)+'> [options]')
else:
    command=sys.argv.pop(1)
    importlib.import_module(COMMANDS[command]).main()
