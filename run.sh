#!/bin/sh

ALMC_DIR="./libs"
LIB_DIR="$ALMC_DIR"/lib
ALMC_JAR="$ALMC_DIR"/ModelChecker.jar


# Command to launch Java
if [ "$ALMC_JAVA" = "" ]; then
	if [ -x /usr/libexec/java_home ]; then
		ALMC_JAVA=`/usr/libexec/java_home`"/bin/java"
	else
		ALMC_JAVA=java
	fi
fi

echo "$ALMC_JAVA"

export DYLD_LIBRARY_PATH="$LIB_DIR"
"$ALMC_JAVA" -Djava.library.path="$LIB_DIR" -jar "$ALMC_JAR" "$@"
