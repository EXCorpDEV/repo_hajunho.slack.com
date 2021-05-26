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

-(DDTBT_ATCH_FILE_DTIL *) SelectImageFileInformation:(NSString *) param
{
    
    NSString *query = [NSString stringWithFormat:@"select a.seq, a.nm_logi_file, a.nm_phys_file, a.yn_mbil_add from ddtbt_atch_file_dtil2 a \
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
                NSLog(@"SelectImageFileInformation %@ %ld %@ %@", atch.nm_logi_file, (long)atch.seq, atch.nm_phys_file, atch.yn_mbil_add);
                sqlite3_finalize(compiledStatement);
                [self checkBackDB];
                return atch;
            }
        }
        sqlite3_finalize(compiledStatement);
        [self checkBackDB];
    } else {
        [self checkBackDB];
    }
    return nil;
}

@end
