class Quality{
    public:
	// default constructor
        Quality(int row, int col, int ldn, float* target_mat, float* baseline_mat);

	void get_minmax(float* x, float& max, float& min);
        float average(float* mat);
        float sdev(float* mat);
        float covariance();

	// main APIs
	float rmse(int verbose);
        float error_rate(int verbose);
        float error_percentage(int verbose);
        float ssim(int verbose);
        float pnsr(int verbose);

    private:
	int row;
	int col;
	int ldn;
	float* target_mat;
	float* baseline_mat;
};
