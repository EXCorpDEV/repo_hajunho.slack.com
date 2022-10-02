// webpack.config.js

const port = process.env.PORT || 3000;

module.exports = {
    mode: 'development', // 1
    entry: './src/index.js', // 2
    output: { // 3
        filename: 'bundle.[hash].js' // 4
    },
    module: {
        rules: [
            { // 1
                test: /\.(js|jsx)$/,
                exclude: /node_modules/,
                use: {
                    loader: 'babel-loader',
                },
            },
            { // 2
                test: /\.html$/,
                use: [
                    {
                        loader: 'html-loader',
                        options: {
                            minimize: true,
                        },
                    },
                ],
            },
        ],
    },
    plugins: [
        new HtmlWebpackPlugin({
            template: 'public/index.html',
        })
    ],
    devServer: {
        host: 'localhost',
        port: port,
        open: true,
    },
};
