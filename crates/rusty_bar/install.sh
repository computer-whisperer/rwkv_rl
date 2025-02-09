#!/bin/sh
BAR_PATH="/home/christian/.local/state/Beyond All Reason/engine/rel2501.2025.01.6"
RUSTY_BAR_PATH="$BAR_PATH/AI/Skirmish/RustyBar/0.1"
mkdir -p "$RUSTY_BAR_PATH"
cp target/debug/librusty_bar.so "$RUSTY_BAR_PATH/libSkirmishAI.so"
cp data/* "$RUSTY_BAR_PATH/"