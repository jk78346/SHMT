class Quality{
    public:
	Quality(int m, int n, int ldn, float* x, float* y); // default constructor

        float average(int m, int n, int ldn, float* x);
        float sdev(int m, int n, int ldn, float* x, float ux);
        float covariance(int m, int n, int ldn, float* x, float* y, float ux, float uy);
        float rmse(int m, int n, int ldn, float* x, float* y, int verbose);
        float error_rate(int m, int n, int ldn, float* x, float* y, int verbose);
        float error_percentage(int m, int n, int ldn, float* x, float* y, int verbose);
        float ssim(int m, int n, int ldn, float* x, float* y, int verbose);
        float pnsr(int m, int n, int ldn, float* x, float* y, int verbose);
	
    private:
	int m;
	int n;
	int ldn;
	float* x;
	float* y;
	
	float ux;
	float uy;
};
