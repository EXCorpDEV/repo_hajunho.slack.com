//
//  sql3hjh.m
//
//  Created by Junho HA on 2021/05/26.
//  Copyright © 2021 hajunho.com All rights reserved.
//

#import <Foundation/Foundation.h>
#import "sql3hjh.h"
#import "FileManager.h"
#import "Entity.h"

@implementation sql3hjh

@synthesize databasePath;

- (id) init
{
    if (self = [super init]) {
        NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,NSUserDomainMask, YES);
        NSString *documentsDirectory = [paths objectAtIndex:0];
        databasePath = [documentsDirectory stringByAppendingPathComponent:@"mbass4.db"];
    }
    return self;
}

-(void) checkBackDB {
    if(database != nil)  {
        if(sqlite3_open([databasePath UTF8String], &database) == SQLITE_OK) {
            sqlite3_finalize;
            sqlite3_close(database);
        } else {
            sqlite3_finalize;
            sqlite3_close(database);
        }
    }
}

+ (void) checkAndCreateDatabase:(BOOL)flag {
    NSString *databaseName = @"mbass4.db";
    NSArray *documentPaths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDir = [documentPaths objectAtIndex:0];
    NSString *databasePath = [documentsDir stringByAppendingPathComponent:databaseName];
    BOOL success;
    // Create a FileManager object, we will use this to check the status
    // of the database and to copy it over if required
    NSFileManager *fileManager = [NSFileManager defaultManager];
    // Check if the database has already been created in the users filesystem
    success = [fileManager fileExistsAtPath:databasePath];
    // If the database already exists then return without doing anything
    
    if(success && flag) return;
    
    // If not then proceed to copy the database from the application to the users filesystem
    // Get the path to the database in the application package
    NSString *databasePathFromApp = [[[NSBundle mainBundle] resourcePath] stringByAppendingPathComponent:databaseName];
    
    // Copy the database from the package to the users filesystem
    [fileManager removeItemAtPath:databasePath error:nil];
    [fileManager copyItemAtPath:databasePathFromApp toPath:databasePath error:nil];
    NSLog(@"databasePathFromApp %@ To %@", databasePathFromApp, databasePath);
}

-(NSMutableArray *) SelectImageFileInformation:(NSString *) param
{
    NSMutableArray *array = [[NSMutableArray alloc] init];
    
    NSString *query = [NSString stringWithFormat:@"select a.seq, a.nm_logi_file, a.nm_phys_file, a.yn_mbil_add, b.nm_tppg from ddtbt_atch_file_dtil2 a \
                       inner join ddtbt_tppg b on a.id_atch_file=b.id_dwg_atch_file \
                       where b.cd_tppg='%@';"
                       ,param
                       ];
    
    NSLog(@" hjhImageFileTitledMapReturn query = %@", query);
    
    [self checkBackDB];
    
    if (sqlite3_open([databasePath UTF8String], &database) == SQLITE_OK) {  //성공적으로 열림
        const char *sqlStatement = [query cStringUsingEncoding:NSUTF8StringEncoding];
        sqlite3_stmt *compiledStatement;
        if(sqlite3_prepare_v2(database, sqlStatement, -1, &compiledStatement, NULL) == SQLITE_OK) {
            // Loop through the results and add them to the feeds array
            while(sqlite3_step(compiledStatement) == SQLITE_ROW) {
                DDTBT_ATCH_FILE_DTIL *atch = [[DDTBT_ATCH_FILE_DTIL alloc] init];
                atch.seq = sqlite3_column_int(compiledStatement, 0);
                atch.nm_logi_file = [NSString stringWithUTF8String:(char *)sqlite3_column_text(compiledStatement, 1)];
                atch.nm_phys_file = [NSString stringWithUTF8String:(char *)sqlite3_column_text(compiledStatement, 2)];
                atch.yn_mbil_add = [NSString stringWithUTF8String:(char *)sqlite3_column_text(compiledStatement, 3)];
                atch.nm_tppg = [NSString stringWithUTF8String:(char *)sqlite3_column_text(compiledStatement, 4)];
                NSLog(@"SelectImageFileInformation %@ %ld %@ %@", atch.nm_logi_file, (long)atch.seq, atch.nm_phys_file, atch.yn_mbil_add);
//                sqlite3_finalize(compiledStatement);
//                [self checkBackDB];
//                return atch;
                [array addObject:atch];
            }
        }
        sqlite3_finalize(compiledStatement);
        [self checkBackDB];
    } else {
        [self checkBackDB];
    }
//    return nil;
    return array;
}

- (void) selectAtch_Common:(NSMutableArray *)array withMode:(NSString *)mode maxLimit:(NSInteger)count {
    [array removeAllObjects];
    
    [self checkBackDB];
    if(sqlite3_open([databasePath UTF8String], &database) == SQLITE_OK) {
        // Setup the SQL Statement and compile it for faster access
        
        NSString *query = [NSString stringWithFormat:@"select a.id_atch_file \
                           ,a.nm_logi_file \
                           ,a.nm_phys_file \
                           ,b.id_dfct \
                           ,b.cd_site \
                           ,b.id_rgst \
                           ,b.mode \
                           ,a.seq \
                           from ddtbt_atch_file_dtil a \
                           inner join ddtbt_dfct b on b.ID_DWG_IMG_ATCH_FILE=a.id_atch_file and b.mode like '%@%%' \
                           where a.yn_svr_trsm='N' \
                           limit %d"
                           ,mode
                           ,count
                           ];
        
        const char *sqlStatement = [query cStringUsingEncoding:NSASCIIStringEncoding];
        sqlite3_stmt *compiledStatement;
        if(sqlite3_prepare_v2(database, sqlStatement, -1, &compiledStatement, NULL) == SQLITE_OK) {
            // Loop through the results and add them to the feeds array
            while(sqlite3_step(compiledStatement) == SQLITE_ROW) {
                
                NSMutableDictionary *dic = [[NSMutableDictionary alloc] initWithCapacity:0];
                [dic setValue:[NSString stringWithUTF8String:(char *)sqlite3_column_text(compiledStatement, 0)] forKey:@"id_atch_file"];
                
                NSString *path = [FileManager getDirImage];
                NSString *filePath = [path stringByAppendingPathComponent:[NSString stringWithUTF8String:(char *)sqlite3_column_text(compiledStatement, 1)]];
                
                NSData *data = [NSData dataWithContentsOfFile:filePath];
                
                [dic setValue:data forKey:@"file"];
                
                /*
                 filePath = [path stringByAppendingPathComponent:[NSString stringWithUTF8String:(char *)sqlite3_column_text(compiledStatement, 2)]];
                 [dic setValue:filePath forKey:@"nm_phys_file"];
                 */
                
                [dic setValue:[NSString stringWithFormat:@"%d", sqlite3_column_int(compiledStatement, 3)] forKey:@"id_dfct"];
                [dic setValue:[NSString stringWithUTF8String:(char *)sqlite3_column_text(compiledStatement, 4)] forKey:@"cd_site"];
                [dic setValue:[NSString stringWithUTF8String:(char *)sqlite3_column_text(compiledStatement, 5)] forKey:@"id_rgst"];
                
                [dic setValue:[NSString stringWithUTF8String:(char *)sqlite3_column_text(compiledStatement, 6)] forKey:@"work_type"];
                
                [dic setValue:[NSString stringWithUTF8String:(char *)sqlite3_column_text(compiledStatement, 7)] forKey:@"seq"];
                
                [array addObject:dic];
            }
        }
        // Release the compiled statement from memory
        sqlite3_finalize(compiledStatement);
        
    }
    [self checkBackDB];
}


- (BOOL) updateAtchYnSvrTrsm_common:(NSString *)atchId seq:(NSInteger)seq {
    NSLog(@"- (BOOL) updateAtchYnSvrTrsm:(NSString *)atchId seq:(NSInteger)seq {");
    [self checkBackDB];
    sqlite3_stmt *statement;
    BOOL bRet = YES;
    
    if (sqlite3_open([databasePath UTF8String], &database) == SQLITE_OK) {
        
        NSString *query = [NSString stringWithFormat:@"update ddtbt_atch_file_dtil set yn_svr_trsm='Y' where id_atch_file='%@' and seq=%d;"
                           ,atchId
                           ,seq
                           ];
        
        const char *insert_stmt = [query UTF8String];
        sqlite3_prepare_v2(database, insert_stmt, -1, &statement, NULL);
        
        int rnt = sqlite3_step(statement);
        
        if (rnt == SQLITE_DONE) {
            
            NSLog(@"데이터 저장 성공-8");
        }
        else {
            NSLog(@"데이터 저장 실패-8");
            //bRet = NO;
        }
        sqlite3_finalize(statement);
    }
    [self checkBackDB];
    return bRet;
}

- (void) selectAtchCommon:(NSMutableArray *)array withMode:(NSString *)mode maxLimit:(NSInteger)count {
    [array removeAllObjects];
    
    [self checkBackDB];
    if(sqlite3_open([databasePath UTF8String], &database) == SQLITE_OK) {
        NSString *query = [NSString stringWithFormat:@"select a.id_atch_file \
                           ,a.nm_logi_file \
                           ,a.nm_phys_file \
                           ,b.id_dfct \
                           ,b.cd_site \
                           ,b.id_rgst \
                           ,b.mode \
                           ,a.seq \
                           from ddtbt_atch_file_dtil a \
                           inner join ddtbt_dfct b on (b.id_wrk_fmer_pic_atch_file=a.id_atch_file or b.id_wrk_aftr_pic_atch_file=a.id_atch_file OR b.ID_DWG_IMG_ATCH_FILE=a.id_atch_file) and b.mode like '%@%%' \
                           where a.yn_svr_trsm='N' AND a.yn_mbil_add='C' \
                           limit %d"
                           ,mode
                           ,count
                           ];
        
        NSLog(@"selectAtch query : %@", query);
        
        const char *sqlStatement = [query cStringUsingEncoding:NSASCIIStringEncoding];
        sqlite3_stmt *compiledStatement;
        if(sqlite3_prepare_v2(database, sqlStatement, -1, &compiledStatement, NULL) == SQLITE_OK) {
            // Loop through the results and add them to the feeds array
            while(sqlite3_step(compiledStatement) == SQLITE_ROW) {
                
                NSMutableDictionary *dic = [[NSMutableDictionary alloc] initWithCapacity:0];
                [dic setValue:[NSString stringWithUTF8String:(char *)sqlite3_column_text(compiledStatement, 0)] forKey:@"id_atch_file"];
                
                NSString *path = [FileManager getDirImage];
                NSString *filePath = [path stringByAppendingPathComponent:[NSString stringWithUTF8String:(char *)sqlite3_column_text(compiledStatement, 1)]];
                
                NSData *data = [NSData dataWithContentsOfFile:filePath];
                
                [dic setValue:data forKey:@"file"];
                
                /*
                 filePath = [path stringByAppendingPathComponent:[NSString stringWithUTF8String:(char *)sqlite3_column_text(compiledStatement, 2)]];
                 [dic setValue:filePath forKey:@"nm_phys_file"];
                 */
                
                [dic setValue:[NSString stringWithFormat:@"%d", sqlite3_column_int(compiledStatement, 3)] forKey:@"id_dfct"];
                [dic setValue:[NSString stringWithUTF8String:(char *)sqlite3_column_text(compiledStatement, 4)] forKey:@"cd_site"];
                [dic setValue:[NSString stringWithUTF8String:(char *)sqlite3_column_text(compiledStatement, 5)] forKey:@"id_rgst"];
                
                [dic setValue:[NSString stringWithUTF8String:(char *)sqlite3_column_text(compiledStatement, 6)] forKey:@"work_type"];
                
                [dic setValue:[NSString stringWithUTF8String:(char *)sqlite3_column_text(compiledStatement, 7)] forKey:@"seq"];
                
                [array addObject:dic];
            }
        }
        // Release the compiled statement from memory
        sqlite3_finalize(compiledStatement);
        
    }
    [self checkBackDB];
}


@end
