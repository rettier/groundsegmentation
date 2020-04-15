#!/bin/bash
cd $(dirname $0)
npm run docs:build
rm -rf ../assets
mv docs/.vuepress/dist/* ../

