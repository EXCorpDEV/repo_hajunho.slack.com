// webpack.config.js
module.exports = {
  mode: 'development', // 1
  entry: './src/index.js', // 2
  output: { // 3
    filename: 'bundle.[hash].js' // 4
  },
};
