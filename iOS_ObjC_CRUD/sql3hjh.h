//
//  sql3hjh.h
//
//  Created by Junho HA on 2021/05/26.
//  Copyright © 2021 hajunho.com All rights reserved.
//

#ifndef sql3hjh_h
#define sql3hjh_h

#import <sqlite3.h>
#import "Entity.h"

@interface sql3hjh : NSObject
{
    sqlite3 *database;
}

@property (nonatomic, strong, readwrite) NSString *databasePath;

-(DDTBT_ATCH_FILE_DTIL *) SelectImageFileInformation:(NSString *) param;

- (id) init;
- (void) checkBackDB;
+ (void) checkAndCreateDatabase:(BOOL)flag;

@end

#endif /* sql3hjh_h */
