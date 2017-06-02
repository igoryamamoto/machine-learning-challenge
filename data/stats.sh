file_name=$1
echo page,M,F > sub_$file_name/stats
echo total,$(grep \"M\" -c $file_name),$(grep \"F\" -c $file_name) >> sub_$file_name/stats
echo brand_landing,$(grep \"M\" -c sub_$file_name/brand_landing),$(grep \"F\" -c sub_$file_name/brand_landing) >> sub_$file_name/stats
echo cart,$(grep \"M\" -c sub_$file_name/cart),$(grep \"F\" -c sub_$file_name/cart) >> sub_$file_name/stats
echo category,$(grep \"M\" -c sub_$file_name/category),$(grep \"F\" -c sub_$file_name/category) >> sub_$file_name/stats 
echo checkout,$(grep \"M\" -c sub_$file_name/checkout),$(grep \"F\" -c sub_$file_name/checkout) >> sub_$file_name/stats 
echo confirmation,$(grep \"M\" -c sub_$file_name/confirmation),$(grep \"F\" -c sub_$file_name/confirmation) >> sub_$file_name/stats
echo home,$(grep \"M\" -c sub_$file_name/home),$(grep \"F\" -c sub_$file_name/home) >> sub_$file_name/stats 
echo product,$(grep \"M\" -c sub_$file_name/products),$(grep \"F\" -c sub_$file_name/products) >> sub_$file_name/stats 
echo purchase,$(grep \"M\" -c sub_$file_name/purchase),$(grep \"F\" -c sub_$file_name/purchase) >> sub_$file_name/stats 
echo search,$(grep \"M\" -c sub_$file_name/search),$(grep \"F\" -c sub_$file_name/search) >> sub_$file_name/stats 
echo subcategory,$(grep \"M\" -c sub_$file_name/subcategory),$(grep \"F\" -c sub_$file_name/subcategory) >> sub_$file_name/stats
