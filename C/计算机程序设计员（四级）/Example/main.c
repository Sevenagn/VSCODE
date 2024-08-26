#include <stdio.h>
#include <stdlib.h>

/* run this program using the console pauser or add your own getch, system("pause") or input loop */

void write_file(){
	FILE *fp=fopen("C:/Users/17167/Documents/ZZP/example.txt","w+");
	srand((unsigned int)time(NULL));
	int num;
	char buf[100];
	int i;
	for (i=0;i<1000;i++){
		num=rand()%100;
		sprintf(buf,"%d\n",num);
		fputs(buf,fp);
	}
	fclose(fp);
} 

void read_file(){
	FILE *fp=fopen("C:/Users/17167/Documents/ZZP/example.txt","a+");
	char buf[100];
	int a[2000];
	int i;
	int num;
	while(1){ 
		fgets(buf,sizeof(buf),fp);
		if(feof(fp)){
			break; 
		}
		
		sscanf(buf,"%d\n",&num);
		
		printf("%s",buf);
		
		a[i] = num;
		i++;
	}
	
	
	int j;
	float sum=0; 
	int len=1000;//sizeof(a) / sizeof(a[0]);
	for(j=0;j<len;j++){
		sum+=a[j];
	}
	float avg;
	avg = sum/1000 ;
	
	int x,y,tmp;
	for(x=0;x<999;x++){
		for(y=0;y<999 - x;y++){
			if(a[y]>a[y+1]){
				tmp=a[y];
				a[y]=a[y+1];
				a[y+1]=tmp;
			}	
		}
	}
	printf("count:%d\n",len);
	printf("sum:%f\n",sum);
	printf("avg:%f\n",avg);
	printf("max:%d\n",a[999]);
	printf("min:%d\n",a[0]);
	fprintf(fp,"count:%d\n",len) ;
	fprintf(fp,"sum:%f\n",sum) ;
	fprintf(fp,"avg:%f\n",avg) ;
	fprintf(fp,"max:%d\n",a[999]) ;
	fprintf(fp,"min:%d\n",a[0]) ;
	
	fclose(fp);
}

int main(int argc, char *argv[]) {
	write_file();
	read_file();
	return 0;
}
