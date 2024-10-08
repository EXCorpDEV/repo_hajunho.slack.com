//
//  swift_settingsTests.swift
//  swift_settingsTests
//
//  Created by Junho HA on 2018. 9. 27..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import XCTest
@testable import swift_settings

class swift_settingsTests: XCTestCase {

    override func setUp() {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        GlobalSettings.shared.logLevel = .all
        
        XCTAssert(GlobalSettings.shared.logLevel.contains(.critical))
        XCTAssert(GlobalSettings.shared.logLevel.contains(.major))
        XCTAssert(GlobalSettings.shared.logLevel.contains(.minor))
        XCTAssert(GlobalSettings.shared.logLevel.contains(.callBack))
        XCTAssert(GlobalSettings.shared.logLevel.contains(.infiniteLoop))
        XCTAssert(GlobalSettings.shared.logLevel.contains(.resourceLeak))
        XCTAssert(GlobalSettings.shared.logLevel.contains(.memoryJobs))
        XCTAssert(GlobalSettings.shared.logLevel.contains(.library))
        
        GlobalSettings.shared.logLevel = .critical
        XCTAssertFalse(GlobalSettings.shared.logLevel.contains(.library))
        XCTAssertTrue(GlobalSettings.shared.logLevel.contains(.critical))
    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

}
