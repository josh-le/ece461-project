# runnning the syntax checker
## usage
ensure you have set up the [github token](#github-token) in the .env file, then do this:
```bash
cd syntax-checker # make sure to cd into this directory first, otherwise it won't find the SYNTAX_CHECKER_ENV file
./<script>.sh
```
## env vars
there is a file here called SYNTAX_CHECKER_ENV that has contains all non-secret env vars for the scripts to work. the scripts all source it at the top, so to get it to work you just have to be in the `syntax-checker` directory for it to work properly.
### github token
we each have one of these. to get it to work with my scripts, export it in a .env file in the __root of the project directory__, __NOT__ the directory where you are running the scripts, with the name `GH_TOKEN`:
```bash
# .env
export GH_TOKEN="sk....token goes here...."
```
## scripts
- `register.sh` - registers us with the syntax checker, should not have to be used again
- `schedule-run.sh` - schedules a run with the autograder. i believe it gets our code by checking out the most recent commit on master, but not 100% sure
- `monitor-runs.sh` - returns a list of all jobs scheduled with the autograder right now
- `best-run.sh` - returns results from our best run
- `last-run.sh` - returns results from our most recent run
- `get-log.sh` - downloads a log file. __you must provide the path as an argument__
