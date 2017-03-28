Create a new directory called tsp_project containing a copy of the repo:
```
  git clone https://github.com/kevinburleigh75/tsp_project.git
```

Check what's changed on the current branch:
```
  git status
```

Add files/directories to staging, but don't commit yet:
```
  git add [file] [dir] ...
```

Commit the changes in staging:
```
  git commit -m 'some useful message'
```

Create a new branch:
```
  git checkout -b [branch name]
```

Merge a branch into the current branch:
```
  git merge [branch name]
```

Delete a branch:
```
  git branch -d [branch name]  (if already merged / empty)
  git branch -D [branch name]  (to force deletion - could result in loss of work)
```

To see a history of commits:
```
  git log
```

There are some more advanced commands, like rebase, that we _might_ want to use,
but for now there are sufficient, IMO.
