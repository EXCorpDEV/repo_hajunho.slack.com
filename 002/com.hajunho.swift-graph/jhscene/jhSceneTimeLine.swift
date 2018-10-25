//
//  jhSceneTimeLine.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 24..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

class jhSceneTimeLine : jhScene {
    
    private var mPanels : [jhPanel<jhSceneTimeLine>] = Array<jhPanel>()
    
    var currentTime: Date = Date(timeIntervalSince1970: 1531180740) //TODO: should be changed to init() creates with current time.
    
    var endTime: Date {
        get {
            return Date(timeIntervalSince1970: currentTime.timeIntervalSince1970 - Double(GS.shared.sceneWidthByTime))
        }
    }
    
    override func createPanels(s : jhScene, withHeightRatios: ratioNtype...) {
        
        var panel : jhPanel<jhSceneTimeLine>? = nil
        var y : CGFloat = 0.0
        var vHeight : CGFloat = 0.0
        
        "".pwd(self)
        
        for rnt in withHeightRatios {
            if(GS.shared.logLevel.contains(.graphPanel)) { print("createPanels(withHeightRatios: CGFloat...)", rnt)}
            
            assert(!(rnt.ratio < 0.1 || rnt.ratio > 10.0), "heightRation Range is 0.1~10.0")
            
            vHeight = rnt.ratio * 0.1 * self.jhSceneFrameHeight
            panel = jhGraphBuilder<jhSceneTimeLine>()
                .type(rnt.type)
                .frame(0, y, jhSceneFrameWidth*4, vHeight)
                .scene(self)
                .build()
            y += vHeight
            
            if GS.shared.logLevel.contains(.graphPanel) {
                print("jhScene_addPanel_mHeightStack =", vHeight, "\n y = \(y) heightRatio = \(rnt)")
            }
            panel!.backgroundColor = UIColor.white
            mPanels.append(panel!)
            panel = nil
        }
    }
    
    var xRatioByTime : CGFloat {
        get {
            return jhDraw.maxR / GS.shared.sceneWidthByTime
        }
    }
    
    override func jhColor(r:CGFloat , g:CGFloat , b:CGFloat , a:Float) -> CGColor {
        return  UIColor(red: r / 255.0, green: g / 255.0, blue: b / 255.0, alpha: r).cgColor
    }
    
    override func drawScene() {
        drawPanels()
    }
    
    override func drawPanels() {
        for x in mPanels {
            super.addSubview(x)
        }
    }
}
